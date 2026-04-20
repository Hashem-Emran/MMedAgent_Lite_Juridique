.PHONY: all download fr ar stratify build notebook test clean

all: build

download:
	python -m dataset.download_jur

fr: download
	python -m dataset.preprocess_jur_fr

ar: download
	python -m dataset.preprocess_jur_ar

stratify: fr ar
	python -m dataset.stratify_jur

build: stratify
	python -m dataset.build_hf_dataset

notebook: build
	jupyter nbconvert --to notebook --execute notebooks/rapport_jurida.ipynb --output rapport_jurida.ipynb

test:
	pytest tests/ -v

clean:
	rm -rf data/raw data/interim data/jurida_processed figures/*.png
