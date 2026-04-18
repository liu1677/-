#include "stm32f10x.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

/* --- 宏定义 --- */
#define BIN_PKG_LEN 10
#define PWM_PRESCALER    71      
#define PWM_PERIOD       19999   
#define STEPS_PER_DEGREE  4.44f 
#define MOVE_DELAY_MS     500    
#define SMOOTH_STEPS      20     

/* --- 全局变量定义 --- */
uint8_t G_BinBuffer[BIN_PKG_LEN];
volatile uint8_t G_PacketReady = 0;
uint8_t BinIndex = 0;

float Current_Angle_A = 90.0f; 
float Current_Angle_B = 90.0f; 
float Current_Angle_C = 90.0f; 
float Current_Angle_D = 90.0f; 
float Current_Angle_E = 90.0f; 
float Current_Angle_F = 60.0f;  

float POS_DROP_YELLOW[6] = {45.0f, 120.0f, 90.0f, 90.0f, 90.0f, 90.0f}; 
float POS_DROP_GREEN[6]  = {135.0f, 120.0f, 90.0f, 90.0f, 90.0f, 90.0f}; 

/* --- 函数原型声明 --- */
void RCC_Configuration(void);
void GPIO_Configuration(void);
void TIM2_PWM_Configuration(void); 
void TIM3_PWM_Configuration(void); 
void USART3_Configuration(void);
void Set_Servo_Angle(TIM_TypeDef* TIMx, uint8_t Channel, float angle);
void Smooth_Set_Servo_Angle(TIM_TypeDef* TIMx, uint8_t Channel, float current_angle, float target_angle);
void Smooth_Set_Stepper_Angle(float current_angle, float target_angle);
void Delay_us(uint32_t us);
void Delay_ms(uint32_t ms);
void Go_To_Home_Position(void);
void Execute_Color_Task(uint8_t *pkg);
void Move_All_Joints(float a, float b, float c, float d, float e, float f); // 新增

/* --- 串口重定向 printf --- */
int fputc(int ch, FILE *f) {
    while (USART_GetFlagStatus(USART3, USART_FLAG_TXE) == RESET);
    USART_SendData(USART3, (uint8_t)ch);
    return ch;
}

/* --- 主函数 --- */
int main(void) {
    RCC_Configuration();
    GPIO_Configuration();
    TIM2_PWM_Configuration();
    TIM3_PWM_Configuration();
    USART3_Configuration();
    
    printf("\r\nSystem initialized. Ready for Hex commands...\r\n");
    Go_To_Home_Position();
	Delay_ms(1000);
Move_All_Joints(69,150,80,95,90,114);
	Move_All_Joints(100,90,90,90,90,114);
	Move_All_Joints(45,120,90,90,90,90);
	Delay_ms(1000);
	Go_To_Home_Position();
	//Move_All_Joints(100,90,90,90,90,90);


    while (1) {
        
			
    }
}

/* --- 核心业务逻辑：分类抓取（已修改 C/E 轴通道） --- */
void Execute_Color_Task(uint8_t *pkg) {
    uint8_t color = pkg[2]; 
    float t_a = (float)pkg[3];
    float t_b = (float)pkg[4];
    float t_c = (float)pkg[5];
    float t_d = (float)pkg[6];
    float t_e = (float)pkg[7];
    float t_f = (float)pkg[8];

    printf("Recv CMD -> Color:%c, A:%.1f, B:%.1f, C:%.1f, D:%.1f, E:%.1f, F:%.1f\r\n", 
           color, t_a, t_b, t_c, t_d, t_e, t_f);

    if (color != 'Y' && color != 'G') {
        printf("Ignore: Color %c is not in target list.\r\n", color);
        return; 
    }

    // 1. 移动到物体上方
    Smooth_Set_Stepper_Angle(Current_Angle_A, t_a);
    Smooth_Set_Servo_Angle(TIM3, 1, Current_Angle_B, t_b);
    // 修改：C 轴现在使用 TIM2 CH2
    Smooth_Set_Servo_Angle(TIM2, 2, Current_Angle_C, t_c);
    // 修改：E 轴现在使用 TIM2 CH1 和 CH4
    Smooth_Set_Servo_Angle(TIM2, 1, Current_Angle_E, t_e);
    Smooth_Set_Servo_Angle(TIM2, 4, Current_Angle_E, t_e);
    Smooth_Set_Servo_Angle(TIM2, 3, Current_Angle_D, t_d);
    Delay_ms(800);

    // 2. 闭合爪子
    Smooth_Set_Servo_Angle(TIM3, 2, Current_Angle_F, 114.0f); 
    Current_Angle_F = 114.0f;
    Delay_ms(500);

    // 3. 移动到指定卸料点
    float *drop = (color == 'Y') ? POS_DROP_YELLOW : POS_DROP_GREEN;
    printf("Moving to Drop Point for %s...\r\n", (color == 'Y') ? "YELLOW" : "GREEN");
    
    Smooth_Set_Stepper_Angle(Current_Angle_A, drop[0]);
    Smooth_Set_Servo_Angle(TIM3, 1, Current_Angle_B, drop[1]);
    // 修改：C 轴使用 TIM2 CH2
    Smooth_Set_Servo_Angle(TIM2, 2, Current_Angle_C, drop[2]);
    // 修改：E 轴使用 TIM2 CH1 和 CH4
    Smooth_Set_Servo_Angle(TIM2, 1, Current_Angle_E, drop[4]);
    Smooth_Set_Servo_Angle(TIM2, 4, Current_Angle_E, drop[4]);
    Smooth_Set_Servo_Angle(TIM2, 3, Current_Angle_D, drop[3]);
    Delay_ms(1000);

    // 4. 松开爪子
    Smooth_Set_Servo_Angle(TIM3, 2, Current_Angle_F, 60.0f);
    Current_Angle_F = 60.0f;
    Delay_ms(500);

    // 5. 复位
    Go_To_Home_Position();
}

/* --- 串口中断（不变） --- */
void USART3_IRQHandler(void) {
    if(USART_GetITStatus(USART3, USART_IT_RXNE) != RESET) {
        uint8_t Res = USART_ReceiveData(USART3);
        
        if (BinIndex == 0 && Res != 0x2C) return; 
        if (BinIndex == 1 && Res != 0x12) { BinIndex = 0; return; }
        
        G_BinBuffer[BinIndex++] = Res;
        
        if(BinIndex >= BIN_PKG_LEN) {
            if(G_BinBuffer[BIN_PKG_LEN - 1] == 0x5B) {
                G_PacketReady = 1; 
            }
            BinIndex = 0; 
        }
    }
}

/* --- 同时移动所有关节（已修改 C/E 通道） --- */
void Move_All_Joints(float a, float b, float c, float d, float e, float f) {
    Smooth_Set_Stepper_Angle(Current_Angle_A, a);
		Delay_ms(500);
    Smooth_Set_Servo_Angle(TIM3, 1, Current_Angle_B, b);
	Delay_ms(500);
    // 修改：C 轴使用 TIM2 CH2
    Smooth_Set_Servo_Angle(TIM2, 2, Current_Angle_C, c);
	Delay_ms(500);
    // 修改：E 轴使用 TIM2 CH1 和 CH4
    Smooth_Set_Servo_Angle(TIM2, 4, Current_Angle_E, e);
	Delay_ms(500);
    Smooth_Set_Servo_Angle(TIM2, 3, Current_Angle_D, d);
	Delay_ms(500);
    Smooth_Set_Servo_Angle(TIM3, 2, Current_Angle_F, f);
	Delay_ms(500);
    
    Current_Angle_A = a;
    Current_Angle_B = b;
    Current_Angle_C = c;
    Current_Angle_D = d;
    Current_Angle_E = e;
    Current_Angle_F = f;
}

/* --- 底层驱动函数（部分修改通道说明） --- */
void Smooth_Set_Servo_Angle(TIM_TypeDef* TIMx, uint8_t Channel, float current_angle, float target_angle) {
    if(target_angle < 0) target_angle = 0;
    if(target_angle > 180) target_angle = 180;
    float angle_diff = target_angle - current_angle;
    if(fabs(angle_diff) < 0.1f) {
        Set_Servo_Angle(TIMx, Channel, target_angle);
        return;
    }
    float step_angle = angle_diff / SMOOTH_STEPS;
    for(int i = 1; i <= SMOOTH_STEPS; i++) {
        Set_Servo_Angle(TIMx, Channel, current_angle + step_angle * i);
        Delay_ms(50);  
    }
}

void Smooth_Set_Stepper_Angle(float current_angle, float target_angle)
{
    if(target_angle < 0) target_angle = 0;
    if(target_angle > 180) target_angle = 180;
    
    float angle_diff = target_angle - current_angle;
    if(fabs(angle_diff) < 0.5f) return;

    // 方向逻辑：与第二段代码完全一致（正转低电平，反转高电平）
    if(angle_diff > 0)
        GPIO_ResetBits(GPIOA, GPIO_Pin_9);   // 正转：方向脚低电平
    else
        GPIO_SetBits(GPIOA, GPIO_Pin_9);     // 反转：方向脚高电平

    int total_steps = (int)(fabs(angle_diff) * STEPS_PER_DEGREE);
    if(total_steps == 0) total_steps = 1;
    
    int steps_per_segment = total_steps / SMOOTH_STEPS;
    if(steps_per_segment == 0) steps_per_segment = 1;
    
    for(int segment = 0; segment < SMOOTH_STEPS; segment++)
    {
        int steps_this_segment = (segment == SMOOTH_STEPS - 1) ? 
                                 (total_steps - steps_per_segment * (SMOOTH_STEPS - 1)) : 
                                 steps_per_segment;
        
        for(int i = 0; i < steps_this_segment; i++)
        {
            GPIO_SetBits(GPIOA, GPIO_Pin_10);   // 步进脉冲高电平
            Delay_us(5000);
            GPIO_ResetBits(GPIOA, GPIO_Pin_10); // 步进脉冲低电平
            Delay_us(5000);
        }
        
        Delay_ms(30);   // 段间停顿，实现平滑效果
    }
    
    Current_Angle_A = target_angle;
}
void Go_To_Home_Position(void) {
    Smooth_Set_Stepper_Angle(Current_Angle_A, 90.0f);
	 Delay_ms(100); 
    Smooth_Set_Servo_Angle(TIM3, 1, Current_Angle_B, 90.0f);
	 Delay_ms(100); 
    // 修改：C 轴使用 TIM2 CH2
    Smooth_Set_Servo_Angle(TIM2, 2, Current_Angle_C, 90.0f);
	 Delay_ms(100); 
    // 修改：E 轴使用 TIM2 CH1 和 CH4
    Smooth_Set_Servo_Angle(TIM2, 1, Current_Angle_E, 90.0f);
	 Delay_ms(100); 
    Smooth_Set_Servo_Angle(TIM2, 4, Current_Angle_E, 90.0f);
	 Delay_ms(100); 
    Smooth_Set_Servo_Angle(TIM2, 3, Current_Angle_D, 90.0f);
	 Delay_ms(100); 
    Smooth_Set_Servo_Angle(TIM3, 2, Current_Angle_F, 60.0f);
    
    Current_Angle_A=90.0f; Current_Angle_B=90.0f; Current_Angle_C=90.0f;
    Current_Angle_D=90.0f; Current_Angle_E=90.0f; Current_Angle_F=60.0f;
}

void Set_Servo_Angle(TIM_TypeDef* TIMx, uint8_t Channel, float angle) {
    uint16_t pulse = 500 + (uint16_t)(angle * 2000.0f / 180.0f);
    if(Channel==1) TIMx->CCR1=pulse; 
    else if(Channel==2) TIMx->CCR2=pulse;
    else if(Channel==3) TIMx->CCR3=pulse; 
    else if(Channel==4) TIMx->CCR4=pulse;
}

void Delay_us(uint32_t count) { for(uint32_t i=0; i<count*8; i++) __NOP(); }
void Delay_ms(uint32_t ms) { for(uint32_t i=0; i<ms; i++) Delay_us(1000); }

/* --- RCC 配置（不变） --- */
void RCC_Configuration(void) {
    RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOA | RCC_APB2Periph_GPIOB | RCC_APB2Periph_AFIO, ENABLE);
    RCC_APB1PeriphClockCmd(RCC_APB1Periph_TIM2 | RCC_APB1Periph_TIM3 | RCC_APB1Periph_USART3, ENABLE);
}

/* --- GPIO 配置（步进电机已改 PA9/PA10，其余不变） --- */
void GPIO_Configuration(void) {
    GPIO_InitTypeDef GPIO_InitStructure;

    // PWM 输出引脚：PA0(TIM2_CH1), PA1(TIM2_CH2), PA2(TIM2_CH3), PA3(TIM2_CH4), PA6(TIM3_CH1), PA7(TIM3_CH2)
    GPIO_InitStructure.GPIO_Pin = GPIO_Pin_0 | GPIO_Pin_1 | GPIO_Pin_2 | GPIO_Pin_3 | GPIO_Pin_6 | GPIO_Pin_7;
    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF_PP;
    GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
    GPIO_Init(GPIOA, &GPIO_InitStructure);

    // 步进电机控制引脚：PA9(方向), PA10(脉冲)
    GPIO_InitStructure.GPIO_Pin = GPIO_Pin_9 | GPIO_Pin_10;
    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;
    GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
    GPIO_Init(GPIOA, &GPIO_InitStructure);

    // 串口3 TX：PB10
    GPIO_InitStructure.GPIO_Pin = GPIO_Pin_10;
    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF_PP;
    GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
    GPIO_Init(GPIOB, &GPIO_InitStructure);

    // 串口3 RX：PB11
    GPIO_InitStructure.GPIO_Pin = GPIO_Pin_11;
    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_IN_FLOATING;
    GPIO_Init(GPIOB, &GPIO_InitStructure);
}

/* --- TIM2 PWM 配置（引脚说明已更新） --- */
void TIM2_PWM_Configuration(void) {
    TIM_TimeBaseInitTypeDef T; 
    TIM_OCInitTypeDef O;
    T.TIM_Period = PWM_PERIOD; 
    T.TIM_Prescaler = PWM_PRESCALER; 
    T.TIM_ClockDivision = 0; 
    T.TIM_CounterMode = TIM_CounterMode_Up;
    TIM_TimeBaseInit(TIM2, &T);
    
    O.TIM_OCMode = TIM_OCMode_PWM1; 
    O.TIM_OutputState = TIM_OutputState_Enable; 
    O.TIM_Pulse = 1500; 
    O.TIM_OCPolarity = TIM_OCPolarity_High;
    
    // CH1: PA0  (现用于 E 轴)
    // CH2: PA1  (现用于 C 轴)
    // CH3: PA2  (D 轴)
    // CH4: PA3  (现用于 E 轴，与 CH1 同角度)
    TIM_OC1Init(TIM2, &O); 
    TIM_OC2Init(TIM2, &O); 
    TIM_OC3Init(TIM2, &O); 
    TIM_OC4Init(TIM2, &O);
    
    TIM_Cmd(TIM2, ENABLE);
}

/* --- TIM3 PWM 配置（不变） --- */
void TIM3_PWM_Configuration(void) {
    TIM_TimeBaseInitTypeDef T; 
    TIM_OCInitTypeDef O;
    T.TIM_Period = PWM_PERIOD; 
    T.TIM_Prescaler = PWM_PRESCALER; 
    T.TIM_ClockDivision = 0; 
    T.TIM_CounterMode = TIM_CounterMode_Up;
    TIM_TimeBaseInit(TIM3, &T);
    
    O.TIM_OCMode = TIM_OCMode_PWM1; 
    O.TIM_OutputState = TIM_OutputState_Enable; 
    O.TIM_Pulse = 1500; 
    O.TIM_OCPolarity = TIM_OCPolarity_High;
    
    // CH1: PA6 (B 轴)
    // CH2: PA7 (F 轴 - 爪子)
    TIM_OC1Init(TIM3, &O); 
    TIM_OC2Init(TIM3, &O);
    
    TIM_Cmd(TIM3, ENABLE);
}

/* --- USART3 配置（不变） --- */
void USART3_Configuration(void) {
    USART_InitTypeDef U; 
    NVIC_InitTypeDef N;
    U.USART_BaudRate = 115200; 
    U.USART_WordLength = USART_WordLength_8b; 
    U.USART_StopBits = USART_StopBits_1;
    U.USART_Parity = USART_Parity_No; 
    U.USART_HardwareFlowControl = USART_HardwareFlowControl_None;
    U.USART_Mode = USART_Mode_Rx | USART_Mode_Tx; 
    USART_Init(USART3, &U);
    USART_ITConfig(USART3, USART_IT_RXNE, ENABLE);
    
    N.NVIC_IRQChannel = USART3_IRQn; 
    N.NVIC_IRQChannelPreemptionPriority = 1; 
    N.NVIC_IRQChannelSubPriority = 1; 
    N.NVIC_IRQChannelCmd = ENABLE;
    NVIC_Init(&N); 
    USART_Cmd(USART3, ENABLE);
}