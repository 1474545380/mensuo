#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <fcntl.h>
#include <unistd.h> 
#include <sys/mman.h>
#include <sys/time.h>
#include "gpio.h"

unsigned char *map_base=NULL;
int dev_fd;


int gpio_init(void)
{	
	dev_fd = open("/dev/mem", O_RDWR | O_SYNC);      
	if (dev_fd < 0)  
	{
		printf("\nopen(/dev/mem) failed.\n");    
		return -1;
	}
	map_base=(unsigned char *)mmap(0,MAP_SIZE,PROT_READ|PROT_WRITE,MAP_SHARED,dev_fd,REG_BASE );
	return 0;

}
int gpio_enable(int gpio_num,int val)
{	int offset,gpio_move;                    
	if(gpio_num > 31) 
	{
		offset = 4;
		gpio_move = gpio_num- 32;
	} 
	else 
	{
		offset = 0;
		gpio_move = gpio_num;
	}
	if(val==0)
	{
		*(volatile unsigned int *)(map_base + GPIO_EN  +offset) |= (1<<gpio_move);         //GPIO使能in
		//printf("Enable GPIO%d in\n",gpio_num);
	}
	else
	{
		*(volatile unsigned int *)(map_base + GPIO_EN  +offset) &= ~(1<<gpio_move);         //GPIO使能out
		//printf("Enable GPIO%d out\n",gpio_num);
	}

	return 0;
}
int gpio_close(void)
{	
  
	if (dev_fd < 0)  
	{
		printf("\nopen(/dev/mem) failed.\n");    
		return -1;
	}

	munmap(map_base,MAP_SIZE);//解除映射关系
	if(dev_fd)
	{
		close(dev_fd);
	}
	return 0;
}



int gpio_write(int gpio_num, int val)
{
	int offset, gpio_move;

	if(gpio_num > 31) 
	{
		offset = 4;
		gpio_move = gpio_num- 32;
	} 
	else 
	{
		offset = 0;
		gpio_move = gpio_num;
	}
	if(val == 1)
	{
		*(volatile unsigned int *)(map_base + GPIO_OUT +offset) |= (1<<gpio_move);     //输出高
	}
	else
	{
		*(volatile unsigned int *)(map_base + GPIO_OUT +offset) &= ~(1<<gpio_move);    //输出底
	}
}


int gpio_read(int gpio_num)
{
	int offset, gpio_move;

	if(gpio_num > 31) 
	{
		offset = 4;
		gpio_move = gpio_num - 32;
	} 
	else 
	{
		offset = 0;
		gpio_move = gpio_num;
	}

	return (*(volatile unsigned int *)(map_base + GPIO_IN +offset) >> gpio_move) & 0x01;     //读取
}





/****************************************************************************** 
 * 延时程序 
 */
struct timeval start_time , end_time;

void delay_us(unsigned int us)
{	
	int i = 0;
	gettimeofday(&start_time,NULL);
	end_time=start_time;
	while(1)
	{	
		if((end_time.tv_usec-start_time.tv_usec)<us)
		{
			gettimeofday(&end_time,NULL);
			i++;
		}
		else
		{
			break;
		}
		if(i>1000)
		{
			break;	
		}
		
	}
}

/*
 * 延时 1ms
 *
 * TODO 换算 TICKS_PER_SECOND
 */
void delay_ms(unsigned int ms)
{
	int i = 0;
	int us = ms*1000;
	gettimeofday(&start_time,NULL);
	end_time=start_time;
	while(1)
	{	
		if((end_time.tv_usec-start_time.tv_usec)<us)
		{
			gettimeofday(&end_time,NULL);
			i++;
		}
		else
		{
			break;
		}
		if(i>1000000)
		{
			break;	
		}
		
	}
}
