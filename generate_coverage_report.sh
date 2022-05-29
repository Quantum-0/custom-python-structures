python3 -m mypy --html-report=mypy-report src
coverage run -m unittest tests/**.py
if (( $? == 0 ))
then
  coverage report -m --omit="tests/*"
  coverage html --omit="tests/*"
  xdg-open htmlcov/index.html
else
  python3 -m pytest -v tests/**.py --html=pytest_report.html --self-contained-html
  xdg-open pytest_report.html
  sleep 10
  rm pytest_report.html
fi