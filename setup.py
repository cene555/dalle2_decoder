from setuptools import setup

setup(
    name="dalle2_decoder",
    packages=[
        "glide_text2im",
        "glide_text2im.clip",
        "glide_text2im.tokenizer",
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
        "blobfile",
        "mpi4py",
    ],
    author="OpenAI",
)
