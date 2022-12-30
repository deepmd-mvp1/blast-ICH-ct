FROM nvcr.io/nvidia/pytorch:20.10-py3

RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
COPY blast_ct /home/blast_ct/
COPY templates /home/templates/
COPY App.py /home
RUN mkdir /home/input
RUN mkdir /home/output
COPY requirements.txt /home
RUN pip3 install -r /home/requirements.txt
RUN cd /home
WORKDIR /home
ENV FLASK_APP=App.py
# RUN  /home/pipeline.sh 
CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]