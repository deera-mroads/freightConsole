from typing import List, Dict, Any
import json, os, base64, re, logging
from collections import defaultdict
from mistralai import Mistral, ImageURLChunk, TextChunk
from pathlib import Path
import mimetypes, fitz
from pdf2image import convert_from_path
from dotenv import load_dotenv
from datetime import datetime
import mysql.connector
from mysql.connector import Error

api_key = os.getenv("MISTRAL_OCR_API_KEY")
client = Mistral(api_key=api_key)

load_dotenv(dotenv_path=".env")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_images_from_pdf(pdf_path, output_folder="extracted_images"):
    """Extract images from PDF pages and rotate them 90 degrees counterclockwise"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    images = convert_from_path(pdf_path)
    
    image_paths = []
    for i, image in enumerate(images[1:], start=2):
        rotated_image = image.rotate(-90, expand=True)
        image_path = f"{output_folder}/page_{i+1}.png"
        rotated_image.save(image_path, "PNG")
        image_paths.append(image_path)
    
    return image_paths

def expand_plz_ranges(data):
    expanded_data = []
    for row in data:
        plz = row["PLZ"]
        if " - " in plz or "‐" in plz:
            match = re.findall(r"\d{2}", plz)
            if len(match) >= 2:
                start, end = int(match[0]), int(match[1])
                for i in range(start, end + 1):
                    new_row = row.copy()
                    new_row["PLZ"] = f"{i:02}..."
                    expanded_data.append(new_row)
        else:
            expanded_data.append(row)
    return expanded_data

def extract_tables(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = "\n".join([page.get_text() for page in doc])
    lines = [line.strip() for line in full_text.split("\n") if line.strip()]
    doc.close()

    table1, table2 = [], []
    for i in range(len(lines)):
        if re.match(r"\d{2}…", lines[i]) and i + 7 < len(lines):
            try:
                if lines[i+1].isdigit() and all(re.search(r'\d+,?\d*\s*€', lines[i+j]) for j in range(2, 8)):
                    entry = {
                        "PLZ": lines[i],
                        "ZONE": lines[i+1],
                        "-50": lines[i+2],
                        "-100": lines[i+3],
                        "-200": lines[i+4],
                        "-300": lines[i+5],
                        "-400": lines[i+6],
                        "-500": lines[i+7],
                    }
                    table1.append(entry)
            except IndexError:
                break

    for i in range(len(lines)):
        if re.match(r"\d{2}…", lines[i]) and i + 6 < len(lines):
            try:
                if lines[i+1].isdigit() and all(re.search(r'\d+,?\d*\s*€', lines[i+j]) for j in range(2, 7)):
                    entry = {
                        "PLZ": lines[i],
                        "ZONE": lines[i+1],
                        "-600": lines[i+2],
                        "-700": lines[i+3],
                        "-800": lines[i+4],
                        "-900": lines[i+5],
                        "-1000": lines[i+6],
                    }
                    table2.append(entry)
            except IndexError:
                break

    return expand_plz_ranges(table1), expand_plz_ranges(table2)

def structured_img_ocr(image_path: str) -> List[Dict[str, Any]]:
    image_file = Path(image_path)
    assert image_file.is_file(), f"The provided image path '{image_path}' does not exist."

    mime_type, _ = mimetypes.guess_type(image_path)
    assert mime_type is not None, f"Could not determine MIME type for '{image_path}'"

    encoded_image = base64.b64encode(image_file.read_bytes()).decode()
    base64_data_url = f"data:{mime_type};base64,{encoded_image}"

    image_response = client.ocr.process(
        document=ImageURLChunk(image_url=base64_data_url),
        model="mistral-ocr-latest"
    )

    image_ocr_markdown = image_response.pages[0].markdown    
    result = []
    lines = image_ocr_markdown.strip().split('\n')
    data_rows = [line for line in lines if line.startswith('|') and '---' not in line]
    
    if len(data_rows) > 1:
        data_rows = data_rows[1:]
    
    headers = ["kunde",
               "unsere_pos", 
               "tag", 
               "abhol_oder_zustellfirma", 
               "plz", 
               "ort", 
               "colli", 
               "kg", 
               "vol", 
               "ihre_pos", 
               "preis", 
               "extras", 
               "extras", 
               "gesamt"]
    
    for row in data_rows:
        cells = [cell.strip() for cell in row.split('|')]
        cells = [cell for cell in cells if cell]
        
        if len(cells) >= len(headers):
            row_data = {}
            for i, header in enumerate(headers):
                if i < len(cells):
                    value = cells[i].strip()
                    if value in ["-", "", "null"]:
                        row_data[header] = None
                    else:
                        if header == "colli" and value:
                            try:
                                row_data[header] = int(value)
                            except ValueError:
                                row_data[header] = value
                        elif header == "kg" and value:
                            try:
                                kg_str = value.replace(',', '.')
                                row_data[header] = float(kg_str)
                            except ValueError:
                                row_data[header] = value
                        elif header == "vol" and value:
                            try:
                                vol_str = value.replace(',', '.')
                                row_data[header] = float(vol_str)
                            except ValueError:
                                row_data[header] = value
                        elif header in ["preis", "extras", "gesamt"] and value:
                            row_data[header] = re.sub(r'(\$)?\s*(\d+,\d+)\s*€\$', r'\2 €', value)
                        else:
                            row_data[header] = value
                else:
                    row_data[header] = None
            
            result.append(row_data)
    
    try:
        chat_response = client.chat.parse(
            model="pixtral-12b-latest",
            messages=[
                {
                    "role": "user",
                    "content": [
                        ImageURLChunk(image_url=base64_data_url),
                        TextChunk(text=(
                            f"This image contains a table with logistics data, in markdown format:\n{image_ocr_markdown}. Please extract the table data only and output as JSON.\n\n"
                            "Format each row as an object with these exact field names:\n"
                            "- kunde\n"
                            "- unsere_pos\n"
                            "- tag\n" 
                            "- abhol_oder_zustellfirma\n"
                            "- plz\n"
                            "- ort\n"
                            "- colli\n"
                            "- kg\n"
                            "- vol\n"
                            "- ihre_pos\n"
                            "- preis\n"
                            "- extras\n"
                            "- extras\n"
                            "- gesamt\n\n"
                            "For numeric fields, convert 'colli' to integers, and 'kg' and 'vol' to floating point numbers with decimal points (not commas).\n"
                            "For currency fields, keep the original format with Euro symbol.\n"
                            "If a field is empty or contains just a dash, set the value to null.\n"
                            "Return just a JSON array of objects where each object is a row from the table, with no extra explanations.\n"
                        ))
                    ]
                }
            ],
            response_format={"type": "json_object"},
            temperature=0
        )
        print("Chat response:", chat_response, ":::::::::::::::")
        api_result = chat_response.choices[0].message.content
        
        if api_result and api_result != "[]":
            try:
                if isinstance(api_result, str):
                    parsed_data = json.loads(api_result)
                else:
                    parsed_data = api_result
                
                if isinstance(parsed_data, dict):
                    for key in ["rows", "data", "entries", "table"]:
                        if key in parsed_data and isinstance(parsed_data[key], list):
                            parsed_data = parsed_data[key]
                            break
                
                if isinstance(parsed_data, list) and len(parsed_data) > 0:
                    return parsed_data
            except Exception as e:
                print(f"Error with API parsing: {e}")
        
        print("Using manual fallback parsing")
        return result    
    except Exception as e:
        print(f"Error with API: {e}, using manual fallback parsing")
        return result
    
def load_json_file(filename):
    """Load and parse a JSON file"""
    with open(filename, 'r', encoding='utf-8') as file:
        return json.load(file)

def clean_price(price_str):
    """Convert price string like '168,00 €' to a float value 168.0"""
    if not price_str or not isinstance(price_str, str):
        return None
    
    if price_str.strip().lower() == 'all in':
        return None
    
    price_cleaned = re.sub(r'[^\d,.]', '', price_str)
    price_cleaned = price_cleaned.replace(',', '.')
    
    try:
        return float(price_cleaned)
    except (ValueError, TypeError):
        return None

def get_weight_range_column(vol):
    """Determine the appropriate weight range column based on volume"""
    weight_ranges = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    
    for weight in weight_ranges:
        if vol <= weight:
            return f"-{weight}"
    
    return "-1000"

def analyze_pricing():
    """Analyze pricing based on PLZ and volume"""
    try:
        price_ranges = load_json_file('cargo7_jobs.json')
        shipment_data = load_json_file('cargo7_priceTable.json')
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return []
    
    shipments = []
    for batch in shipment_data:
        for entry in batch:
            if isinstance(entry, dict) and 'plz' in entry:
                shipments.append(entry)

    pricing_dict = {}
    for entry in price_ranges:
        plz_pattern = entry['PLZ'].rstrip('.…')
        if '+' in plz_pattern:
            prefixes = [p.strip() for p in plz_pattern.split('+')]
        else:
            prefixes = [plz_pattern]
        
        for prefix in prefixes:
            clean_prefix = re.match(r'(\d+)', prefix)
            if clean_prefix:
                pricing_dict[clean_prefix.group(1)] = entry

    results = []
    for shipment in shipments:
        plz = shipment.get('plz', '')
        plz_prefix = plz[:2] if plz else ''
        kg = shipment.get('kg', 0)
        vol = shipment.get('vol', 0)
        
        if isinstance(kg, str):
            try:
                kg = float(kg.replace('.', '').replace(',', '.'))
            except ValueError:
                kg = 0

        if isinstance(vol, str):
            try:
                vol = float(vol.replace('.', '').replace(',', '.'))
            except ValueError:
                vol = 0
        
        measure = max(kg, vol)
        weight_range = get_weight_range_column(measure)
        actual_price = clean_price(shipment.get('gesamt', '0'))

        if actual_price is None:
            actual_price = clean_price(shipment.get('preis', '0'))

        if plz_prefix in pricing_dict:
            expected_price = clean_price(pricing_dict[plz_prefix].get(weight_range, '0'))
            if actual_price is not None and expected_price is not None:
                match_status = 'MATCH' if abs(actual_price - expected_price) < 0.01 else 'NO MATCH'
            else:
                match_status = 'INVALID_PRICE'

            results.append({
                'unsere_pos': shipment.get('unsere_pos', ''),
                'abhol_oder_zustellfirma': shipment.get('abhol_oder_zustellfirma', ''),
                'plz': plz,
                'plz_prefix': plz_prefix,
                'vol': vol,
                'kg': kg,
                'weight_range': weight_range,
                'actual_price': actual_price,
                'expected_price': expected_price,
                'status': match_status,
                'extras': shipment.get('extras', ''),
                'zone': pricing_dict[plz_prefix].get('ZONE', '')
            })
        else:
            results.append({
                'unsere_pos': shipment.get('unsere_pos', ''),
                'abhol_oder_zustellfirma': shipment.get('abhol_oder_zustellfirma', ''),
                'plz': plz,
                'plz_prefix': plz_prefix,
                'vol': vol,
                'kg': kg,
                'weight_range': weight_range,
                'actual_price': actual_price,
                'expected_price': None,
                'status': 'PLZ NOT FOUND',
                'extras': shipment.get('extras', ''),
                'zone': ''
            })
    
    return results

def print_results(results):
    """Print the analysis results in a readable format"""
    print("Price Comparison Analysis:")
    print("-" * 120)
    print(f"{'Position':<15} {'Company':<20} {'PLZ':<8} {'Volume':<10} {'KG':<10} {'Weight Range':<12} {'Actual':<10} {'Expected':<10} {'Status':<15} {'Extras'}")
    print("-" * 120)
    
    for result in results:
        company = result['abhol_oder_zustellfirma']
        if len(company) > 18:
            company = company[:17] + "…"
        
        vol = float(result['vol']) if isinstance(result['vol'], (int, float)) else 0.0
        kg = float(result['kg']) if isinstance(result['kg'], (int, float)) else 0.0
            
        print(f"{result['unsere_pos']:<15} {company:<20} {result['plz']:<8} {vol:<10.1f} "
              f"{kg:<10.1f} {result['weight_range']:<12} "
              f"{result['actual_price'] if result['actual_price'] is not None else 'N/A':<10} "
              f"{result['expected_price'] if result['expected_price'] is not None else 'N/A':<10} "
              f"{result['status']:<15} {result['extras']}")

    match_count = sum(1 for r in results if r['status'] == 'MATCH')
    mismatch_count = sum(1 for r in results if r['status'] == 'MISMATCH')
    not_found_count = sum(1 for r in results if r['status'] == 'PLZ NOT FOUND')
    invalid_price_count = sum(1 for r in results if r['status'] == 'INVALID_PRICE')
    
    print("\nSummary:")
    print(f"Total Jobs: {len(results)}")
    print(f"Matching Prices: {match_count}")
    print(f"Mismatching Prices: {mismatch_count}")
    print(f"PLZ Not Found: {not_found_count}")
    print(f"Invalid Price Data: {invalid_price_count}")
    
def analyze_mismatches(results):
    """Analyze price mismatches to identify patterns"""
    mismatches = [r for r in results if r['status'] == 'MISMATCH']
    
    for m in mismatches:
        price_diff = m['actual_price'] - m['expected_price'] if m['actual_price'] and m['expected_price'] else 'N/A'
        if price_diff != 'N/A':
            diff_percent = (price_diff / m['expected_price']) * 100 if m['expected_price'] else float('inf')
            extras_note = f" (Has extras: {m['extras']})" if m['extras'] and m['extras'] != '€' else ""
            print(f"{m['unsere_pos']} - {m['abhol_oder_zustellfirma']}: Actual: {m['actual_price']:.2f}€, "
                  f"Expected: {m['expected_price']:.2f}€, Diff: {price_diff:.2f}€ ({diff_percent:.1f}%){extras_note}")

def save_results_to_json(results, filename='analysis_results.json'):
    """Save analysis results to a JSON file"""
    output_path = os.path.join(os.path.dirname(__file__), filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")

def create_database_connection():
    """Create a connection to the freight_console MySQL database."""
    try:
        connection = mysql.connector.connect(
            host=os.getenv("host"),
            user=os.getenv("user"),
            password=os.getenv("password"),
            database=os.getenv("database"),
        )
        if connection.is_connected():
            print("Connected to MySQL database")
        return connection
    except Error as e:
        print(f"Error connecting to MySQL: {e}")
        return None

def filter_invalid_entries(data):
    """Filter out entries with specific invalid values."""
    filtered_data = [
        entry for entry in data
        if not (
            entry.get("unsere_pos") == ":--:" and
            entry.get("abhol_oder_zustellfirma") == ":--:" and
            entry.get("plz") == ":--:" and
            entry.get("plz_prefix") == ":-" and
            entry.get("vol") == 0 and
            entry.get("kg") == 0 and
            entry.get("weight_range") == "-50" and
            entry.get("actual_price") is None and
            entry.get("expected_price") is None and
            entry.get("status") == "PLZ NOT FOUND" and
            entry.get("extras") == ":--:" and
            entry.get("zone") == ""
        )
    ]
    return filtered_data

def store_results_in_database(results, pdf_image_path):
    """Store the filtered results in the database."""
    connection = None
    try:
        connection = create_database_connection()
        if not connection:
            raise Exception("Failed to connect to the database.")

        cursor = connection.cursor()
        file_name = os.path.basename(pdf_image_path).split('.')[0]
        results_json = json.dumps(results, ensure_ascii=False)
        created_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        insert_query = """
        INSERT INTO analysis_results (file_name, results, created_date)
        VALUES (%s, %s, %s)
        """
        cursor.execute(insert_query, (file_name, results_json, created_date))
        connection.commit()
        return True

    except Error as e:
        logging.error(f"Error while inserting data into the database: {e}")
        raise
    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()