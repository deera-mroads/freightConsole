from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import shutil, json, os
from utils import (
    extract_images_from_pdf,
    extract_tables,
    structured_img_ocr,
    analyze_pricing,
    filter_invalid_entries,
    store_results_in_database
)
from collections import defaultdict
from typing import Dict, List, Optional

app = FastAPI(title="Freight Console API")

# State management
class ProcessState:
    def __init__(self):
        self.invoice_jobs_path: Optional[str] = None
        self.price_table_path: Optional[str] = None
        self.image_paths: List[str] = []
        self.structured_responses: List[dict] = []
        self.combined_list: List[dict] = []

    def reset(self):
        # Cleanup files
        if self.invoice_jobs_path and os.path.exists(self.invoice_jobs_path):
            os.remove(self.invoice_jobs_path)
        if self.price_table_path and os.path.exists(self.price_table_path):
            os.remove(self.price_table_path)
        for path in self.image_paths:
            if os.path.exists(path):
                os.remove(path)
        
        # Reset state
        self.__init__()

state = ProcessState()

class MessageResponse(BaseModel):
    message: str

@app.post("/invoiceJobs/", response_model=MessageResponse)
async def invoiceJobs(invoice_jobs: UploadFile = File(...)):
    try:
        # Save uploaded file temporarily
        invoice_jobs_path = f"temp_{invoice_jobs.filename}"
        with open(invoice_jobs_path, "wb") as buffer:
            shutil.copyfileobj(invoice_jobs.file, buffer)

        # Process PDF
        image_paths = extract_images_from_pdf(invoice_jobs_path)
        structured_responses = []
        
        for image_path in image_paths:
            structured_response = structured_img_ocr(image_path)
            structured_responses.append(structured_response)

        # Update state
        state.invoice_jobs_path = invoice_jobs_path
        state.image_paths = image_paths
        state.structured_responses = structured_responses

        with open("cargo7_priceTable.json", "w", encoding="utf-8") as f:
            json.dump(structured_responses, f, indent=2, ensure_ascii=False)

        result = MessageResponse(message="Invoice jobs processed successfully")
        print(type(result))
        return result

    except Exception as e:
        state.reset()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/priceTable/")
async def priceTable(price_table_tariff: UploadFile = File(...)):
    try:
        # Save uploaded file temporarily
        price_table_path = f"temp_{price_table_tariff.filename}"
        with open(price_table_path, "wb") as buffer:
            shutil.copyfileobj(price_table_tariff.file, buffer)

        # Process PDF
        table1, table2 = extract_tables(price_table_path)
        combined = defaultdict(dict)

        for row in table1:
            key = (row["PLZ"], row["ZONE"])
            combined[key].update(row)

        for row in table2:
            key = (row["PLZ"], row["ZONE"])
            combined[key].update(row)

        combined_list = list(combined.values())

        # Update state
        state.price_table_path = price_table_path
        state.combined_list = combined_list

        with open("cargo7_jobs.json", "w", encoding="utf-8") as f:
            json.dump(combined_list, f, indent=2, ensure_ascii=False)

        return JSONResponse(content={"message": "Price table processed successfully"})

    except Exception as e:
        state.reset()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyzePricing/")
async def analyze_pricing_endpoint():
    try:
        if not (state.invoice_jobs_path and state.price_table_path):
            raise HTTPException(
                status_code=400, 
                detail="Both invoice jobs and price table must be uploaded first"
            )

        # Analyze and store results
        results = analyze_pricing()
        filtered_results = filter_invalid_entries(results)
        store_results_in_database(filtered_results, state.invoice_jobs_path)

        # Cleanup and reset state
        state.reset()

        return JSONResponse(content={"results": filtered_results})

    except Exception as e:
        state.reset()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run("app:app", port=5000, reload=True, log_level="info")
