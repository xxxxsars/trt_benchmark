FROM nvcr.io/nvidia/tensorrt:21.03-py3

COPY . /tmp/trt_benchmark

RUN apt-get update
RUN mkdir /var/run/sshd

RUN apt-get install -y openssh-server vim protobuf-compiler libprotoc-dev
RUN echo 'root:root' |chpasswd
RUN sed -ri 's/^.*PermitRootLogin\s+.*/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed -ri 's/.*UsePAM yes/#UsePAM yes/g' /etc/ssh/sshd_config

# onnx converter only support tf < 2.4
RUN  pip install tensorflow==2.3.1
RUN  pip install tf2onnx onnx onnxruntime Cython onnxmltools matplotlib
#fix pycuda running error
RUN  pip install --upgrade numpy


EXPOSE 22
#ENTRYPOINT service ssh restart
CMD ["/usr/sbin/sshd", "-D"]