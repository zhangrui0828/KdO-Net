CUR_PATH=$PWD
echo $CUR_PATH

cd PCL_EXE
sudo ./build.sh

#直接在终端执行python demo.py，目前直接执行还存在问题
#cd $CUR_PATH/SmoothNet/
#python demo.py
