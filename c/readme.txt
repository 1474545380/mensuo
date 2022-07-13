gcc -o main main.c gpio.c i2c/*.c src/*.c -lpthread -lm

gcc -fPIC -c SensorControl.c *.c ../gpio.c ../i2c/*.c -lpthread -lm
gcc -shared -o SensorControl.so *.o 

from ctypes import cdll
mydll = cdll.LoadLibrary('/home/loongson/Experimental_project/6.Face_recognition_module/c/src/SensorControl.so')
mydll.Sensor_init()
mydll.Sensor_Control(1)
