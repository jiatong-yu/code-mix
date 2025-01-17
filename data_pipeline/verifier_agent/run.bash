docker rm "d3"
docker build -t data-generator .
docker run -d --name "d3" data-generator
docker logs -f "d3"