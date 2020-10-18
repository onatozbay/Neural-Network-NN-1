/*
MIT License
Copyright (c) 2020 Onat Özbay
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define Ref_No       4
#define Input_No     2
#define Hidden_No    2
#define Output_No    1
#define e            0.1

/*//Alternative function
double Hyper_Tan(double x) {
  return ((exp(2*x)-1)/(exp(2*x)+1));
}

double Hyper_Tan_Deriv(double x) {
  return 1-pow(((exp(2*x)-1)/(exp(2*x)+1)), 2);
}

float Random_No() {
  return ((float)(rand()%200)-100)/(100);
}
*/
double Sigmoid(double x) {
   return (1/(1+exp(-x)));
}

double Sigmoid_Deriv(double x) {
   return x*(1-x);
}

float Random_No() {
   return ((float)(rand()%100))/(100);
}

int main() {
   srand(time(NULL));

   //Make layers and references
   double Hidden_Layer[Hidden_No];
   double Output_Layer[Output_No];

   double Hidden_Bia[Hidden_No];
   double Output_Bia[Output_No];

   double Hidden_Weight[Input_No][Hidden_No];
   double Output_Weight[Hidden_No][Output_No];

   double Ref_Input[Ref_No][Input_No]={{0, 0}, {0, 1}, {1, 0}, {1, 1}};
   double Ref_Output[Ref_No][Output_No]={{0}, {1}, {1}, {0}};

   //Hidden weights random assigned
   for(int i=0; i<Input_No; i++) {
      for(int j=0; j<Hidden_No; j++) {
         Hidden_Weight[i][j]=Random_No();
      }
   }

   //Hidden bias and output weights random assigned
   for(int i=0; i<Hidden_No; i++) {
      Hidden_Bia[i]=Random_No();
      for(int j=0; j<Output_No; j++) {
         Output_Weight[i][j]=Random_No();
      }
   }

   //Output bias random assigned
   for(int i=0; i<Output_No; i++) {
      Output_Bia[i]=Random_No();
   }

   //Learn times
   for(int o=0; o<300000; o++) {
      printf("\n");
      //Each reference times
      for(int g=0; g<Ref_No; g++) {

         //Forward pass from input to hidden layer
         for(int i=0; i<Hidden_No; i++) {
            double Activation=Hidden_Bia[i];
            for(int j=0; j<Input_No; j++) {
               Activation+=Ref_Input[g][j]*Hidden_Weight[j][i];
            }
            Hidden_Layer[i]=Sigmoid(Activation);
         }

      //Forward pass from hidden to output layer
      for(int i=0; i<Output_No; i++) {
        double Activation=Output_Bia[i];
        for(int j=0; j<Hidden_No; j++) {
          Activation+=Hidden_Layer[j]*Output_Weight[j][i];
        }
        Output_Layer[i]=Sigmoid(Activation);
      }

      //Print operands
      printf("Ref Input: [%.2f, %.2f]\t   Output: [%.2f]\tRef Output: [%.2f]\n", Ref_Input[g][0], Ref_Input[g][1], Output_Layer[0], Ref_Output[g][0]);

      //Backprop from output to hidden layer
      double Output_Delta[Output_No];
      for(int i=0; i<Output_No; i++) {
        double Output_Error=Ref_Output[g][i]-Output_Layer[i];
        Output_Delta[i]=Output_Error*Sigmoid_Deriv(Output_Layer[i]);
      }

      //Change output bia and weight
      for(int i=0; i<Output_No; i++) {
        Output_Bia[i]+=Output_Delta[i]*e;
        for(int j=0; j<Hidden_No; j++) {
          Output_Weight[j][i]+=Hidden_Layer[j]*Output_Delta[i]*e;
        }
      }

      //Backprop from hidden to input layer
      double Hidden_Delta[Hidden_No];
      for(int i=0; i<Hidden_No; i++) {
        double Hidden_Error=0;
        for(int j=0; j<Output_No; j++) {
          Hidden_Error+=Output_Delta[j]*Output_Weight[i][j];
        }
        Hidden_Delta[i]=Hidden_Error*Sigmoid_Deriv(Hidden_Layer[i]);
      }

      //Change hidden bia and weight
      for(int i=0; i<Hidden_No; i++) {
        Hidden_Bia[i]+=Hidden_Delta[i]*e;
        for(int j=0; j<Input_No; j++) {
          Hidden_Weight[j][i]+=Ref_Input[g][j]*Hidden_Delta[i]*e;
        }
      }
    } //End of reference times
  } //End of learn times

  //Print final results
  printf("\nFinal Hidden Weights[%.0d][%.0d]={", Input_No, Hidden_No);
  for(int i=0; i<Input_No; i++) {
    for(int j=0; j<Hidden_No-1; j++) {
      if((i==Input_No-1)&&(j==Hidden_No-2)) {
        printf("{%.2f, %.2f}", Hidden_Weight[i][j], Hidden_Weight[i][j+1]);
      }
      else {
        printf("{%.2f, %.2f}, ", Hidden_Weight[i][j], Hidden_Weight[i][j+1]);
      }
    }
  }
  printf("}\n");

  printf("Final Hidden Bias[%.0d]={", Hidden_No);
  for(int i=0; i<Hidden_No; i++) {
    if(i==Hidden_No-1) {
      printf("%.2f", Hidden_Bia[i]);
    }
    else {
      printf("%.2f, ", Hidden_Bia[i]);
    }
  }
  printf("}\n");

  printf("Final Output Weights[%.0d][%.0d]={", Hidden_No, Output_No);
  for(int i=0; i<Hidden_No; i++) {
    for(int j=0; j<Output_No; j++) {
      if((i==Hidden_No-1)&&(j==Output_No-1)) {
        printf("{%.2f}", Output_Weight[i][j]);
      }
      else {
        printf("{%.2f}, ", Output_Weight[i][j]);
      }
    }
  }
  printf("}\n");

  printf("Final Output Bias[%.0d]={", Output_No);
  for(int i=0; i<Output_No; i++) {
    if(Output_No==1) {
      printf("%.2f", Output_Bia[i]);
    }
    else if(i==Output_No-1) {
      printf("{%.2f}", Output_Bia[i]);
    }
    else {
      printf("{%.2f}, ", Output_Bia[i]);
    }
  }
  printf("}\n");
  if((Ref_Output[Ref_No-4][Output_No-1]==0)&&(Ref_Output[Ref_No-3][Output_No-1]==1)&&(Ref_Output[Ref_No-2][Output_No-1]==1)&&(Ref_Output[Ref_No-1][Output_No-1]==1)) {
     printf("Gate Name: OR\n");
  }
  else if((Ref_Output[Ref_No-4][Output_No-1]==1)&&(Ref_Output[Ref_No-3][Output_No-1]==0)&&(Ref_Output[Ref_No-2][Output_No-1]==0)&&(Ref_Output[Ref_No-1][Output_No-1]==0)) {
     printf("Gate Name: NOR\n");
  }
  else if((Ref_Output[Ref_No-4][Output_No-1]==0)&&(Ref_Output[Ref_No-3][Output_No-1]==1)&&(Ref_Output[Ref_No-2][Output_No-1]==1)&&(Ref_Output[Ref_No-1][Output_No-1]==0)) {
     printf("Gate Name: XOR\n");
  }
  else if((Ref_Output[Ref_No-4][Output_No-1]==1)&&(Ref_Output[Ref_No-3][Output_No-1]==0)&&(Ref_Output[Ref_No-2][Output_No-1]==0)&&(Ref_Output[Ref_No-1][Output_No-1]==1)) {
    printf("Gate Name: XNOR\n");
  }
  else if((Ref_Output[Ref_No-4][Output_No-1]==0)&&(Ref_Output[Ref_No-3][Output_No-1]==0)&&(Ref_Output[Ref_No-2][Output_No-1]==0)&&(Ref_Output[Ref_No-1][Output_No-1]==1)) {
    printf("Gate Name: AND\n");
  }
  else if((Ref_Output[Ref_No-4][Output_No-1]==1)&&(Ref_Output[Ref_No-3][Output_No-1]==1)&&(Ref_Output[Ref_No-2][Output_No-1]==1)&&(Ref_Output[Ref_No-1][Output_No-1]==0)) {
    printf("Gate Name: NAND\n");
  }
  else {
    printf("Gate Name: UNKNOWN\n");
  }
} //End of main
