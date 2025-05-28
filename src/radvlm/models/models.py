# controller.py
import subprocess
import os

def run_in_env(env_name, script, input_image_path, input_text):
    try:
        result = subprocess.check_output([
            "conda", "run", "-n", env_name, "python", script, input_image_path, input_text
        ], stderr=subprocess.STDOUT)
        return result.decode()
    except subprocess.CalledProcessError as e:
        return f"Error running {script} in {env_name}:\n{e.output.decode()}"

if __name__ == "__main__":
    input_text = "Generate a radiology report for this X-ray."
    here = os.path.dirname(os.path.abspath(__file__))
    # Build the path to the image in the data folder
    input_image_path = os.path.join(here, "..", "data", "s50010747.jpg")
    input_image_path = os.path.abspath(input_image_path)
    # Build the path to the scripts
    qwen_script_path = os.path.join(here, "qwen25-vl.py")
    deepseek_script_path = os.path.join(here, "deepseek-vl2.py")
    
    # input_image_path = "data/s50010747.jpg"  # Example image path, adjust as needed

    print("Running Qwen...")
    output_qwen = run_in_env("qwenenv", qwen_script_path, input_image_path, input_text)
    print(output_qwen)

    print("Running DeepSeek...")
    output_deepseek = run_in_env("deepseekenv", deepseek_script_path, input_image_path, input_text)
    print(output_deepseek)
