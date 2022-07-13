
#include <stdio.h>

#include "../i2c/smbus.h"
#include "../i2c/i2cbusses.h"
#include "../i2c/util.h"
#include "../gpio.h"
#include "MLX90614.h"

int file;

void MLX90614_init(void)
{
	char filename[20];
	int ID[4];
	file = open_i2c_dev(I2C, filename, sizeof(filename), 0);
	delay_us(100);
}

int MLX90614_GET()
{	
	float Ta_d;
	float To_d;
	float tbody = 0;
	int tbody_100 = 0;
	short To, Ta;
	char datal, datah;
	set_slave_addr(file, MLX90614_DEVICE_ADDR, 1);
	delay_us(100);
	Ta = i2c_smbus_read_word_data(file, 0x06);
	delay_us(100);
	To = i2c_smbus_read_word_data(file, 0x07);
	delay_us(100);

	Ta_d = Ta * 0.02 - 273.15;
	To_d = To * 0.02 - 273.15;
    	tbody = getTempbody(Ta_d ,To_d);
	tbody_100 = tbody*100;
	return tbody_100;

}

float getTempbody(float ta,float tf)
{
    float tbody = 0;
    float tf_low,tf_high = 0;
    float TA_LEVEL = 25;
    
    if(ta <= TA_LEVEL)
    {
        tf_low  = 32.66 + 0.186 * (ta - TA_LEVEL);
        tf_high = 34.84 + 0.148 * (ta - TA_LEVEL);
    }
    else
    {
        tf_low  = 32.66 + 0.086 * (ta - TA_LEVEL);
        tf_high = 34.84 + 0.1 * (ta - TA_LEVEL);
    }
    
    if(tf_low <= tf && tf <= tf_high)
    {
        tbody = 36.3 + 0.5 / (tf_high - tf_low) * (tf - tf_low);
    }
    else if(tf > tf_high)
    {
        tbody = 36.8 + (0.029321 + 0.002364 * ta) * (tf - tf_high);
    }
    else if(tf < tf_low)
    {
        tbody = 36.3 + (0.551658 + 0.021525 * ta) * (tf - tf_low);
    }
    
    return tbody;
}
