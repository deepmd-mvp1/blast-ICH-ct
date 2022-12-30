docker build --no-cache -t anilyerramasu/blast-ich .
docker run   --rm -p 9000:5000 -v $(pwd)/input:/home/input -v $(pwd)/output:/home/output anilyerramasu/segliver_3d_gpu
