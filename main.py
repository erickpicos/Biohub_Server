from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
from pathlib import Path
from datetime import datetime
import pandas as pd
import json
import os
from config import *

from trainer import train_and_export_onnx

app = FastAPI()

DATASET_FOLDER.mkdir(exist_ok=True)
ONNX_FOLDER.mkdir(exist_ok=True)

class BrainBitResistData(BaseModel):
    O1: float
    O2: float
    T3: float
    T4: float
class EEGSample(BaseModel):
    packNum: int
    marker: int
    sampleIndex: int
    timestamp: float
    C1: float
    C2: float
    C3: float
    C4: float
class EEGRecordingWindow(BaseModel):
    label: str
    startIndex: int
    endIndex: int
    startTime: float
    endTime: float
    samples: List[EEGSample]
class EEGTrainingSession(BaseModel):
    userId: int
    resistData: BrainBitResistData

    calibrationSamplesRaw: List[EEGSample]
    calibrationSamplesFiltered: List[EEGSample]

    meanSession: List[float]
    stdSession: List[float]

    eegRecordingWindowsRaw: List[EEGRecordingWindow]
    eegRecordingWindowsFiltered: List[EEGRecordingWindow]
    eegRecordingWindowsZscoreCalibration: List[EEGRecordingWindow]

def save_windows( windows: List[EEGRecordingWindow], folder: Path, session: EEGTrainingSession, calibration_raw: str, calibration_filtered: str ):
    folder.mkdir(parents=True, exist_ok=True)
    saved_files = []

    for window in windows:
        if len(window.samples) != 1000:
            raise HTTPException( status_code=400, detail=f"Window {window.label} does not contain 1000 samples" )

        existing = [ f for f in os.listdir(folder) if f.startswith( f"S{session.userId:03d}_Task{window.label}_" ) and f.endswith(".csv") ]
        next_id = len(existing) + 1
        base_name = f"S{session.userId:03d}_Task{window.label}_{next_id}"

        csv_path = folder / f"{base_name}.csv"
        json_path = folder / f"{base_name}.json"

        # CREATE DATAFRAME
        df = pd.DataFrame([{"C1": s.C1, "C2": s.C2, "C3": s.C3, "C4": s.C4 } for s in window.samples ])
        # SAVE DF
        df.to_csv( csv_path, index=False, float_format="%.10f" )

        metadata = {
            "userId": session.userId,
            "label": window.label,
            "samples": len(window.samples),
            "channels": ["C1", "C2", "C3", "C4"],
            "samplingRate": 250,
            "duration": 4.0,
            "createdAt": datetime.utcnow().isoformat(),
            "startIndex": window.startIndex,
            "endIndex": window.endIndex,
            "startTime": window.startTime,
            "endTime": window.endTime,
            # SESSION NORMALIZATION VALUES
            "meanSession": session.meanSession,
            "stdSession": session.stdSession,
            "resistData": session.resistData.model_dump(),

            "calibrationRawFile": calibration_raw,
            "calibrationFilteredFile": calibration_filtered
        }

        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=4)

        saved_files.append(base_name)

    return saved_files

def save_calibration(samples: List[EEGSample], folder: Path, session: EEGTrainingSession, kind: str):
    folder.mkdir(parents=True, exist_ok=True)

    existing = [
        f for f in os.listdir(folder)
        if f.startswith(f"S{session.userId:03d}_calibration_{kind}_") and f.endswith(".csv")
    ]

    next_id = len(existing) + 1
    base_name = f"S{session.userId:03d}_calibration_{kind}_{next_id}"

    csv_path = folder / f"{base_name}.csv"

    df = pd.DataFrame([
        {"C1": s.C1, "C2": s.C2, "C3": s.C3, "C4": s.C4}
        for s in samples
    ])

    df.to_csv(csv_path, index=False)

    return f"{base_name}.csv"

@app.post("/upload-session")
def upload_session(session: EEGTrainingSession):
    # VALIDATION
    if len(session.eegRecordingWindowsRaw) == 0:
        raise HTTPException(status_code=400, detail="No recording windows received" )

    user_folder = DATASET_FOLDER / f"User_{session.userId}" 
    raw_folder = user_folder / "Raw"
    filtered_folder = user_folder / "Filtered"
    zscore_folder = user_folder / "ZScore"
    calibrationSamples_folder = user_folder / "Calibration"

    #Save calibration samples
    calibration_raw_file = save_calibration( session.calibrationSamplesRaw, calibrationSamples_folder, session, "raw" )
    calibration_filtered_file = save_calibration( session.calibrationSamplesFiltered, calibrationSamples_folder, session, "filtered" )

    # SAVE EEGS
    raw_saved = save_windows( session.eegRecordingWindowsRaw, raw_folder, session, calibration_raw_file, calibration_filtered_file )
    filtered_saved = save_windows( session.eegRecordingWindowsFiltered, filtered_folder, session, calibration_raw_file, calibration_filtered_file )
    zscore_calibration_saved = save_windows( session.eegRecordingWindowsZscoreCalibration, zscore_folder, session, calibration_raw_file, calibration_filtered_file )

    

    return {
        "success": True,
        "rawSaved": raw_saved,
        "filteredSaved": filtered_saved,
        "zscoreCalibrationSaved": zscore_calibration_saved
    }

# TRAIN
@app.post("/train/{user_id}")
def train(user_id: int):
    try:
        onnx_path = train_and_export_onnx(user_id)
        return {
            "success": True,
            "message": "Training completed",
            "model": onnx_path
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

#GET MODEL
@app.get("/model/{user_id}")
def get_model(user_id: int):
    model_path = ( ONNX_FOLDER / f"User_{user_id}" / "model.onnx" )

    if not model_path.exists(): raise HTTPException( status_code=404, detail="Model not found" )

    return FileResponse( path=model_path, filename="model.onnx", media_type="application/octet-stream" )

class InferenceWindow(BaseModel):
    expectedLabel: str
    predictedLabel: str
    probabilities: List[float]
    samples: List[EEGSample]
class InferenceSession(BaseModel):
    userId: int
    windows: List[InferenceWindow]

@app.post("/save-inference-session")
def save_inference_session(session: InferenceSession):
    if len(session.windows) == 0:
        raise HTTPException( status_code=400, detail="No inference windows received" )

    folder = save_inference_session_files(session)

    return {
        "success": True,
        "folder": folder,
        "windows": len(session.windows)
    }

def save_inference_session_files(session: InferenceSession):
    inference_folder = DATASET_FOLDER / f"User_{session.userId}" / "Inference"
    inference_folder.mkdir(parents=True, exist_ok=True)

    existing_sessions = [f for f in os.listdir(inference_folder) if f.startswith("Session_")]
    session_id = len(existing_sessions) + 1

    session_folder = inference_folder / f"Session_{session_id:03d}"
    session_folder.mkdir(parents=True, exist_ok=True)

    metadata_windows = []

    for i, window in enumerate(session.windows, start=1):
        csv_path = session_folder / f"Window_{i:02d}.csv"

        df = pd.DataFrame([{ "C1": s.C1, "C2": s.C2, "C3": s.C3, "C4": s.C4 } for s in window.samples ])
        df.to_csv(csv_path, index=False, float_format="%.10f")

        metadata_windows.append({
            "window": i,
            "expectedLabel": window.expectedLabel,
            "predictedLabel": window.predictedLabel,
            "probabilities": window.probabilities,
            "samples": len(window.samples)
        })

    metadata = {
        "userId": session.userId,
        "createdAt": datetime.utcnow().isoformat(),
        "windows": metadata_windows
    }

    with open(session_folder / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    return str(session_folder)

















