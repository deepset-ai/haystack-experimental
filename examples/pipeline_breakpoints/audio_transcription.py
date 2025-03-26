import argparse
from pathlib import Path

from haystack import component
from haystack.dataclasses.byte_stream import ByteStream
from haystack.components.audio import LocalWhisperTranscriber
from haystack.utils import ComponentDevice

from haystack_experimental.core.pipeline.pipeline import Pipeline

@component
class StreamFetcher:
    @component.output_types(audio_stream=list[ByteStream])
    def run(self, urls: list[str]) -> dict[str, list[ByteStream]]:
        output = {'audio_stream': [ByteStream.from_file_path(Path(f)) for f in urls]}
        return output


def pipeline():
    pipe = Pipeline()
    pipe.add_component("fetcher", StreamFetcher())
    cpu_device = ComponentDevice.from_str("cpu")
    pipe.add_component("transcriber", LocalWhisperTranscriber(model="tiny", device=cpu_device))
    pipe.connect("fetcher.audio_stream", "transcriber.sources")

    return pipe


def breakpoint():
    pipe = pipeline()
    path = Path(__file__).parent / "audio_files"
    urls = [Path(path,"EK_19690725_64kb.mp3"), Path(path,"jfk_1963_0626_berliner_64kb.mp3")]
    _ = pipe.run(data={"fetcher": {"urls": urls}}, breakpoints={("transcriber", 0)})


def resume(resume_state):
    pipe = pipeline()
    resume_state = pipe.load_state(resume_state)
    result = pipe.run(data={}, resume_state=resume_state)
    print(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--breakpoint", action="store_true", help="Run pipeline with breakpoints")
    parser.add_argument("--resume", action="store_true", help="Resume pipeline from a saved state")
    parser.add_argument("--state", type=str, required=False)
    args = parser.parse_args()

    if args.breakpoint:
        breakpoint()

    elif args.resume:
        if args.state is None:
            raise ValueError("state is required when resuming, pass it with --state <state>")
        resume(args.state)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()