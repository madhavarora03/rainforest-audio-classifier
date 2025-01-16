#include <Arduino.h>

int total_loop_runs = 0;

// put function declarations here:
int myFunction(int, int);

void setup()
{
  Serial.begin(115200);
  Serial.printf("Program Setup Start!\n");
  // put your setup code here, to run once:
  int result = myFunction(2, 3);
}

void loop()
{
  // put your main code here, to run repeatedly:
  total_loop_runs++;
  Serial.printf("Program loop execution %d\n", total_loop_runs);
  // Wait 5 seconds before repeat
  delay(5000);
}

// put function definitions here:
int myFunction(int x, int y)
{
  return x + y;
}