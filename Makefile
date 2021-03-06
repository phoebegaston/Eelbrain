
clean:
	rm -rf coverage .coverage build dist

clean-py:
	find . -type f -name *.pyc -delete
	find . -type f -name *.pyo -delete

doc:
	rm -rf doc/build doc/source/generated
	python setup.py build_sphinx

test:
	nosetests eelbrain eelbrain/_stats eelbrain/_wxgui eelbrain/experiment eelbrain/load eelbrain/plot

test-coverage:
	rm -rf coverage .coverage
	nosetests --with-coverage --cover-package=eelbrain --cover-html --cover-html-dir=coverage

pypi: doc
	rm -rf build dist
	python setup.py sdist bdist_wheel bdist_egg upload upload_docs

.PHONY: clean clean-py doc test test-coverage pypi
