import modal

app = modal.App("images-demo")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("pandas==2.2.0", "numpy")
)

@app.function(image=image)
def hello():
    import pandas as pd
    import numpy as np
    df = pd.DataFrame({"a": np.arange(3)})
    return df.to_dict()

@app.local_entrypoint()
def main():
    print(hello.remote())