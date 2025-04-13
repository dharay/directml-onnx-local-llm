import onnxruntime as ort
import onnxruntime_genai as og

# List available execution providers



import numpy as np 
import os
os.environ["PROCESSOR_ARCHITECTURE"] = "AMD64"
print(ort.get_available_providers())

config = og.Config("models\directml-int4-awq-block-128")
print(config)
config.clear_providers()
# if args.execution_provider != "cpu":
#     if args.verbose: print(f"Setting model to {args.execution_provider}")
#     config.append_provider(args.execution_provider)
model = og.Model(config)