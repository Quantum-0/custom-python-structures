coverage run -m unittests tests/*
coverage report -m --omit="tests/*"
coverage html --omit="tests/*"
xdg-open htmlcov/index.html