import argparse
import os


def main():
    parser = argparse.ArgumentParser(
        description="Launch the compression study explorer and intent classifier"
    )
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--artifact", help="Released bundle directory")
    parser.add_argument("--download-model", action="store_true")
    parser.add_argument(
        "--int8", action="store_true", help="Use quantized CPU embedding and linear kernels"
    )
    parser.add_argument("--port", type=int, default=8105)
    args = parser.parse_args()
    if args.demo:
        os.environ["APP_DEMO"] = "1"
    if args.download_model:
        from .artifact import download

        args.artifact = str(download("runtime/banking77-student"))
    if args.artifact:
        os.environ["MODEL_ARTIFACT"] = args.artifact
    if args.int8:
        os.environ["MODEL_INT8"] = "1"
    import uvicorn

    uvicorn.run("smallmodel.api:app", host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
