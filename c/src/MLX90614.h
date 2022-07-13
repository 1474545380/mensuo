#ifndef _MLX90614_H
#define _MLX90614_H

#include<stdint.h>
#define MLX90614_DEVICE_ADDR  0x5A
#define I2C 0

void MLX90614_init(void);
int MLX90614_GET();
float getTempbody(float ta,float tf);
#endif
