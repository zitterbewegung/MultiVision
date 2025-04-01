import pandas as pd
import numpy as np
import io # Required for reading file upload stream
import os # For path joining

# --- FastAPI Imports ---
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
import uvicorn

# --- ChartRecommender Imports ---
# Assuming your project structure allows these imports
# Adjust paths if necessary (e.g., using sys.path.append)
import torch
from model.encodingModel import ScoreNetLSTM, ChartTypeLSTM # Assuming these are the correct classes
from utils.ChartRecommender import ChartRecommender
# from utils.VegaLiteRender import VegaLiteRender # Optional: if you want to return Vega-Lite specs

# --- Constants and Configuration ---
WORD_EMBEDDING_PATH = 'utils/en-50d-200000words.vec'
SINGLE_CHART_MODEL_PATH = 'trainedModel/singleChartModel.pt'
CHART_TYPE_MODEL_PATH = 'trainedModel/chartType.pt'
MV_MODEL_PATH = 'trainedModel/mvModel.pt' # Path for the MV model

# Determine device (use CUDA if available, otherwise CPU)
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# --- Global Variables (Load models once at startup) ---
word_embedding_dict = {}
column_score_model = None
chart_type_model = None
mv_model = None # Variable for the MV model

# --- Helper Function to Load Models ---
def load_models():
    global word_embedding_dict, column_score_model, chart_type_model, mv_model

    # Load Word Embeddings
    print(f"Loading word embeddings from: {WORD_EMBEDDING_PATH}")
    if not os.path.exists(WORD_EMBEDDING_PATH):
        raise FileNotFoundError(f"Word embedding file not found at: {WORD_EMBEDDING_PATH}")
    try:
        with open(WORD_EMBEDDING_PATH) as file_in:
            for idx, line in enumerate(file_in):
                if idx == 0:  # Skip header line if present
                    try:
                        # Check if the first line is header (e.g., "200000 50")
                        int(line.split()[0])
                        int(line.split()[1])
                        continue
                    except (ValueError, IndexError):
                        # If not header, process it as a word
                        pass
                word, *features = line.split()
                try:
                     # Ensure features are floats
                    word_embedding_dict[word] = np.array(features, dtype=np.float32)
                except ValueError as e:
                    print(f"Skipping line {idx+1} due to parsing error: {line[:50]}... Error: {e}")
                    continue
        print(f"Loaded {len(word_embedding_dict)} word embeddings.")
    except Exception as e:
        print(f"Error loading word embeddings: {e}")
        raise

    # Load Single Chart Score Model
    print(f"Loading single chart score model from: {SINGLE_CHART_MODEL_PATH}")
    if not os.path.exists(SINGLE_CHART_MODEL_PATH):
        raise FileNotFoundError(f"Single chart model not found at: {SINGLE_CHART_MODEL_PATH}")
    try:
        # Adjust input_size, seq_length, batch_size based on your actual model definition
        column_score_model = ScoreNetLSTM(input_size=96, seq_length = 4, batch_size=1, pack = True).to(DEVICE) # Use batch_size=1 for inference if 
appropriate
        column_score_model.load_state_dict(torch.load(SINGLE_CHART_MODEL_PATH, map_location=DEVICE))
        column_score_model.eval()
        print("Single chart score model loaded successfully.")
    except Exception as e:
        print(f"Error loading single chart score model: {e}")
        raise

    # Load Chart Type Prediction Model
    print(f"Loading chart type model from: {CHART_TYPE_MODEL_PATH}")
    if not os.path.exists(CHART_TYPE_MODEL_PATH):
         raise FileNotFoundError(f"Chart type model not found at: {CHART_TYPE_MODEL_PATH}")
    try:
        # Adjust input_size, hidden_size, etc. based on your actual model definition
        chart_type_model = ChartTypeLSTM(input_size = 96, hidden_size = 400, seq_length = 4, num_class = 9, bidirectional = True).to(DEVICE)
        chart_type_model.load_state_dict(torch.load(CHART_TYPE_MODEL_PATH, map_location=DEVICE))
        chart_type_model.eval()
        print("Chart type model loaded successfully.")
    except Exception as e:
        print(f"Error loading chart type model: {e}")
        raise

    # Load MV Recommender Model
    print(f"Loading MV recommender model from: {MV_MODEL_PATH}")
    if not os.path.exists(MV_MODEL_PATH):
        raise FileNotFoundError(f"MV model not found at: {MV_MODEL_PATH}")
    try:
         # Adjust input_size, seq_length based on your actual model definition
        mv_model = ScoreNetLSTM(input_size=9, seq_length = 12).to(DEVICE) # Use appropriate parameters
        mv_model.load_state_dict(torch.load(MV_MODEL_PATH, map_location=DEVICE))
        mv_model.eval()
        print("MV recommender model loaded successfully.")
    except Exception as e:
        print(f"Error loading MV recommender model: {e}")
        raise

# --- FastAPI App Initialization ---
app = FastAPI(title="Chart Recommender API", version="1.0")

# --- Load models on startup ---
# Use lifespan manager for robust startup/shutdown model loading (FastAPI >= 0.90)
# For older versions, you might load directly here, but lifespan is preferred.
@app.on_event("startup")
async def startup_event():
    print("Starting up and loading models...")
    try:
        load_models()
        print("Models loaded successfully.")
    except Exception as e:
        print(f"FATAL: Failed to load models during startup: {e}")
        # Depending on desired behavior, you might want to exit or prevent the app from starting fully.
        # For now, we print the error. FastAPI might still start but endpoints will fail.
        # Consider raising the exception again if you want startup to halt.


# --- API Endpoints ---
@app.post("/recommend/")
async def recommend_charts(
    file: UploadFile = File(..., description="CSV file containing the data to recommend charts for."),
    recommend_type: str = Query("single", enum=["single", "mv"], description="Type of recommendation: 'single' chart or 'mv' (multi-view)."),
    max_mv_charts: int = Query(4, ge=1, le=12, description="Maximum number of charts in the recommended MV (only used if recommend_type='mv').")
):
    """
    Recommends charts based on the uploaded CSV data.

    - Upload a CSV file.
    - Specify the recommendation type ('single' or 'mv').
    - If 'mv', specify the maximum number of charts.
    """
    # Ensure models are loaded
    if not all([word_embedding_dict, column_score_model, chart_type_model, mv_model]):
        raise HTTPException(status_code=503, detail="Models are not loaded or failed to load. Check server logs.")

    # Read uploaded file content
    try:
        contents = await file.read()
        # Use io.StringIO to treat the byte string as a file for pandas
        data_stream = io.StringIO(contents.decode('utf-8'))
        df = pd.read_csv(data_stream)
    except pd.errors.ParserError as e:
        raise HTTPException(status_code=400, detail=f"Error parsing CSV file: {e}")
    except UnicodeDecodeError:
         raise HTTPException(status_code=400, detail="Error decoding file. Please ensure it is UTF-8 encoded.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading or processing uploaded file: {e}")
    finally:
        await file.close() # Ensure the file is closed

    if df.empty:
        raise HTTPException(status_code=400, detail="Uploaded CSV file is empty.")

    # Instantiate ChartRecommender with the loaded data and models
    try:
        # Pass the globally loaded models and embeddings
        # Ensure ChartRecommender's __init__ matches these arguments
        chartRecommender = ChartRecommender(df,
                                            word_embedding_dict,
                                            column_score_model,
                                            chart_type_model,)
    except Exception as e:
        # Catch potential errors during ChartRecommender initialization (e.g., data preprocessing)
        raise HTTPException(status_code=500, detail=f"Error initializing ChartRecommender: {e}")


    # Perform recommendation based on type
    try:
        if recommend_type == "single":
            # Get raw chart recommendations (list of dicts)
            charts_data = chartRecommender.charts
            if not charts_data:
                 return {"message": "No single chart recommendations generated.", "recommendations": []}

            # Convert to DataFrame for sorting (as in the notebook)
            recommended_charts_df = pd.DataFrame.from_records(charts_data)

            # Sort by final_score
            recommended_charts_df = recommended_charts_df.sort_values(by='final_score', ascending=False)

            # Convert the sorted DataFrame back to a list of dictionaries for JSON response
            # Handle potential NaN values for JSON compatibility
            recommendations = recommended_charts_df.replace({np.nan: None}).to_dict('records')
            return {"recommendation_type": "single", "recommendations": recommendations}

        elif recommend_type == "mv":
             # Use the recommend_mv method with the globally loaded mv_model
            recommended_mv = chartRecommender.recommend_mv(mv_model,
                                                           current_mv=[], # Start from scratch
                                                           max_charts=max_mv_charts)

            if not recommended_mv:
                 return {"message": "No MV recommendations generated.", "recommendations": []}

            # The output of recommend_mv is expected to be a list of chart dicts
            return {"recommendation_type": "mv", "max_charts_requested": max_mv_charts, "recommendations": recommended_mv}

    except Exception as e:
         # Catch errors during the recommendation process itself
         # Log the detailed error server-side for debugging
         print(f"Error during recommendation: {e}") # Basic logging
         # Consider adding more robust logging
         raise HTTPException(status_code=500, detail=f"An internal error occurred during chart recommendation: {e}")


# --- Root endpoint (optional) ---
@app.get("/")
async def read_root():
    return {"message": "Welcome to the Chart Recommender API. Use the /docs endpoint for API documentation."}


# --- Run the app (for local development) ---
if __name__ == "__main__":
    # Set host to "0.0.0.0" to be accessible from other devices on the network
    # Use reload=True for development to automatically reload on code changes
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
