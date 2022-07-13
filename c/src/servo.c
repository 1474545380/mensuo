
#include "servo.h"
#include "../gpio.h"
#include <stdio.h>
#include "../i2c/util.h"
#include "../i2c/smbus.h"
#include "../i2c/i2cbusses.h"

int file;
void SERVO_init(void)
{
	gpio_init();
	gpio_enable(GPIO7_LED_GRN, 1);
	gpio_enable(GPIO60_LED_GRN, 1);
	gpio_enable(GPIO9_BUZ1, 1);
	char filename[20];
	//打开I2C
	file = open_i2c_dev(I2C, filename, sizeof(filename), 0);
	delay_us(10);
	//设置器件地址
	set_slave_addr(file, GP7101_ADDRESS, 1);
	delay_us(10);
	Set_SERVO_CLOSE();
}


void FS90_Set_PWM(unsigned short brightpercent)
{
	unsigned char data[3] = {0};
	unsigned short brightness = brightpercent;

	if (brightpercent >= 140)
	{
		brightness = 2880 - (brightpercent - 140) * 30; //每减30，指针偏转一度
	}
	else
	{
		brightness = 8350 - brightpercent * 40; //每减40，指针偏转一度
	}

	set_slave_addr(file, GP7101_ADDRESS, 1);
	delay_us(100);
	//16位PWM模式
	data[0] = WR_16BIT_CMD;
	//数据
	data[1] = brightness;
	data[2] = brightness >> 8;
	short DATA = data[1] | (data[2] << 8);
	i2c_smbus_write_word_data(file, 0x02, brightness);
}

void Set_SERVO_OPEN()
{
	FS90_Set_PWM(90);
	gpio_write(GPIO7_LED_GRN, 0);
	gpio_write(GPIO60_LED_GRN, 1);
	gpio_write(GPIO9_BUZ1,1);
}

void Set_SERVO_CLOSE()
{
	FS90_Set_PWM(0);
	gpio_write(GPIO7_LED_GRN, 1);
	gpio_write(GPIO60_LED_GRN, 0);
	gpio_write(GPIO9_BUZ1,0);
}
