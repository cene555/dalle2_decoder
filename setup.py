from setuptools import setup

setup(
    name="dalle2-decoder",
    packages=[
        "dalle2_decoder",
        "dalle2_decoder.clip",
        "dalle2_decoder.tokenizer",
    ],
    package_data={
        "glide_text2im.tokenizer": [
            "bpe_simple_vocab_16e6.txt.gz",
            "encoder.json.gz",
            "vocab.bpe.gz",
        ],
        "glide_text2im.clip": ["config.yaml"],
    },
    install_requires=[
        "Pillow",
        "attrs",
        "torch",
        "filelock",
        "requests",
        "tqdm",
        "ftfy",
        "regex",
        "numpy",
        "mpi4py",
        "blobfile"
    ],
    author="OpenAI",
)
