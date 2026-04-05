Food Freshness Detection - CLI Tool

QUICK START

1. Install requirements
   pip install -r requirements.txt

2. Run interactive menu
   menu.bat

3. Or use direct commands
   python food_cli.py predict --image "path/to/image.jpg"


COMMANDS

1. PREDICT - Single Image Prediction
   python food_cli.py predict --image "path/to/image.jpg"
   
   Example:
   python food_cli.py predict --image "C:\Users\saket\SEM6-PROJECTS\DL\Project\Dataset\test\fresh_apple_1.jpg"
   
   Output:
   Image: fresh_apple_1.jpg
   Prediction: FRESH
   Confidence: 95.23%
     - Fresh: 95.23%
     - Spoiled: 4.77%


2. PREDICT - Batch Processing (Multiple Images)
   python food_cli.py predict --image "path/to/folder" --batch --save
   
   Example:
   python food_cli.py predict --image "C:\path\to\images" --batch --save
   
   This will:
   - Process all images in the folder
   - Display summary (total, fresh%, spoiled%)
   - Save results to CSV in cli_outputs folder


3. LAYERS - Layer-by-Layer Visualization
   python food_cli.py layers --image "path/to/image.jpg"
   
   Shows how image transforms through each neural network layer:
   - Input image shape and stats
   - Layer 1-13: Each layer's output shape, parameters, statistics
   - Final prediction
   
   Example output:
   INPUT IMAGE
     Shape: (1, 3, 224, 224)
     Channels: 3 (RGB)
     Dimensions: 224 x 224
     Device: cpu
     Mean: 0.350234
     Std: 0.215321
     Min: 0.001234
     Max: 0.998765

   1. CONVOLUTION (3->8)
     Shape: (1, 8, 224, 224)
     Parameters: 224
     Mean: 0.125436
     Std: 0.087654
     ...

   FINAL PREDICTION
     Class: FRESH
     Confidence: 95.23%
     Fresh: 95.23%
     Spoiled: 4.77%


4. ANALYZE - Detailed Analysis
   python food_cli.py analyze --image "path/to/image.jpg"
   
   Shows prediction details for a single image.


5. INFO - Model Information
   python food_cli.py info
   
   Displays:
   - Model name
   - Total parameters
   - Architecture details
   - Layer-by-layer breakdown


6. EVALUATE - Evaluate on Test Set
   python food_cli.py evaluate
   
   Tests model on entire test dataset.
   Displays:
   - Accuracy
   - Precision
   - Recall
   - F1-Score


GPU USAGE

Add --gpu flag to use GPU acceleration (if available)

Examples:
python food_cli.py predict --image "image.jpg" --gpu
python food_cli.py layers --image "image.jpg" --gpu
python food_cli.py evaluate --gpu


USING THE MENU

Run: menu.bat

Then select:
1. Quick Prediction (Single Image)
2. Quick Prediction (Multiple Images)  
3. Detailed Analysis
4. Layer-by-Layer Visualization
5. Model Information
6. Evaluate Model on Test Set
7. Exit

Just type the image name (not full path):
Example: fresh_apple_1.jpg


TEST IMAGES AVAILABLE

Available in: Dataset/test/

Fresh foods:
fresh_apple_1.jpg
fresh_banana_2.jpg
fresh_bread_1.jpg
fresh_orange_1.jpg
fresh_mango_1.jpg

Spoiled foods:
moldy_apple_3.jpg
rotten_banana_5.jpg
moldy_bread_1.jpg
expired_yogurt_10.jpg
spoiled_milk_10.jpg

And many more! Use any image name from that folder.


OUTPUT FILES

Batch prediction results are saved as CSV:
Location: Project/cli_outputs/predictions_YYYYMMDD_HHMMSS.csv

CSV columns:
image, prediction, confidence, fresh_probability, spoiled_probability, timestamp


EXAMPLE WORKFLOW

Step 1: Single prediction
python food_cli.py predict --image "C:\Users\saket\SEM6-PROJECTS\DL\Project\Dataset\test\fresh_apple_1.jpg"

Output:
✓ Model loaded successfully
✓ Total parameters: 3,217,554

Image: fresh_apple_1.jpg
Prediction: FRESH
Confidence: 95.23%
  - Fresh: 95.23%
  - Spoiled: 4.77%

Step 2: See layer details
python food_cli.py layers --image "C:\Users\saket\SEM6-PROJECTS\DL\Project\Dataset\test\fresh_apple_1.jpg"

Shows 13 layers and how image transforms through them.

Step 3: Batch process folder
python food_cli.py predict --image "C:\Users\saket\SEM6-PROJECTS\DL\Project\Dataset\test" --batch --save

Output:
Found 500 images
Processing... [##########] 100%

Total: 500
Fresh: 450 (90.0%)
Spoiled: 50 (10.0%)

Results saved to: Project/cli_outputs/predictions_20260406_143022.csv


TROUBLESHOOTING

Error: "Model not found"
Solution: Make sure best_model.pth exists in Project folder

Error: "Image not found"
Solution: Use full path to image file

Error: "No module named 'food_cli'"
Solution: Make sure you're in the Codebase folder
cd c:\Users\saket\SEM6-PROJECTS\DL\Project\Codebase

Error: "Command not recognized"
Solution: Check spelling of command (predict, layers, analyze, info, evaluate)

GPU not detected:
Solution: --gpu flag will fall back to CPU automatically


DIRECT COMMAND EXAMPLES

Quick prediction:
python food_cli.py predict --image "fresh_apple_1.jpg"

Show layers:
python food_cli.py layers --image "moldy_bread_1.jpg"

Model info:
python food_cli.py info

Evaluate model:
python food_cli.py evaluate

Batch with GPU:
python food_cli.py predict --image "./test_images" --batch --save --gpu


SUPPORTED IMAGE FORMATS

jpg, jpeg, png, bmp, tiff


SINGLE FILE ARCHITECTURE

food_cli.py contains:
- Prediction engine
- Layer visualization
- Analysis tools
- Model loading
- Evaluation metrics
- Batch processing

All in one clean file!
