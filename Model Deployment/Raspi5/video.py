import degirum as dg
import degirum_tools

# Configuration for a single video input
model_name = "yolov8n"
video_source = 'test.mp4'  # Path to video file
display_name = "AI Dashcam"

# Inference settings
inference_host_address = "@local"  # or "@local" if running locally
zoo_url = "zoo_url"
token = ''  # Leave empty string '' if running local without token

# Load the model
model = dg.load_model(
    model_name=model_name,
    inference_host_address=inference_host_address,
    zoo_url=zoo_url,
    token=token
)

# Display and infer
with degirum_tools.Display(display_name) as output_display:
    for inference_result in degirum_tools.predict_stream(model, video_source):
        output_display.show(inference_result)

print(f"Video stream '{display_name}' has been processed.")
[Is mit possible to revert the camera before passing to video_source? Other please remain unchange]