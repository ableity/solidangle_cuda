/*
基于孟凡珍师姐的matlab创建立体角系统矩阵程序
v 1.0
不使用cuda，c代码版
注意matlab数组从1开始而c从0开始
by 李蕾
*/
#include <iostream>  
#include <string>
#include "cuda_runtime.h"  
#include "device_launch_parameters.h"  
//#pragma comment(lib,"cudart.lib")
#include "G:\cuda\bitmap\common\cpu_bitmap.h"
#include "G:\cuda\bitmap\common\book.h"
#include <cmath>
using namespace std;

void showarray(double *a, int n)
{
	//输出数组，调试用
	if (n < 6)
	{
		printf("[");
		for (int i = 0; i < n; i++)
		{
			if (i != 0)
				printf(",");
			printf("%.4f", a[i]);
		}
		printf("]");
	}
	else
	{
		printf("[");
		for (int i = 0; i < 3; i++)
		{
			if (i != 0)
				printf(",");
			printf("%.4f", a[i]);
		}
		printf("...");
		for (int i = n - 4; i < n; i++)
		{
			if (i != n - 4)
				printf(",");
			printf("%.4f", a[i]);
		}
		printf("]\n");
	}
}

__global__ void initsinglevalue(double *a, int n, double value)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int tid = blockDim.x * gridDim.x;
	while (i < n)
	{
		a[i] = value;
		i += tid;
	}
}
double* init_array_single_value(int num, double value)
{
	double *in;
	//大数组初始化，采用cuda
	in = (double*)malloc(num*sizeof(double));
	double *in_cuda;
	HANDLE_ERROR(cudaMalloc((void**)&in_cuda, num*sizeof(double)));
	initsinglevalue << <1000, 1000 >> >(in_cuda, num, value);
	HANDLE_ERROR(cudaMemcpy(in, in_cuda, num*sizeof(double), cudaMemcpyDeviceToHost));
	cudaFree(in_cuda);
	return in;
}

__global__ void initmanyvalue(double *a, int n, double begin, double diff)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int tid = blockDim.x * gridDim.x;
	while (i < n)
	{
		a[i] = begin + i * diff;
		i += tid;
	}
}
double* init_array_many_value(int num, double begin, double diff)
{
	double *in;
	//等差数列初始化，采用cuda
	in = (double*)malloc(num*sizeof(double));
	double *in_cuda;
	HANDLE_ERROR(cudaMalloc((void**)&in_cuda, num*sizeof(double)));
	initmanyvalue << <1000, 1000 >> >(in_cuda, num, begin, diff);
	HANDLE_ERROR(cudaMemcpy(in, in_cuda, num*sizeof(double), cudaMemcpyDeviceToHost));
	cudaFree(in_cuda);
	return in;
}

double sum_array(double *a, int n, int m)
{
	//矩阵从n加到m
	double out = 0;
	for (int i = n; i <= m; i++)
	{
		out += a[i];
	}
	return out;
}

double* findCen(double *point,double *LORUp, double kx,double ky, double kz,const int *CrySize, double VoxSize, double Dis,double OffsetUP)
{
	double *centerPoint;
	centerPoint = (double*)malloc(3*sizeof(double));
	for (int i = 0; i < 3; i++)
	{
		centerPoint[i] = point[i];
	}

	double YSide[2] = { LORUp[2 - 1] + (double)CrySize[2 - 1] / 2, LORUp[2 - 1] - CrySize[3 - 1] / 2 };
	double ZSide[2] = { LORUp[3 - 1] + (double)CrySize[3 - 1] / 2, LORUp[3 - 1] - CrySize[2 - 1] / 2 };
	
	double tUp = (point[1 - 1] - Dis / 2 - OffsetUP) / kx;
	double YUp = point[2 - 1] - ky * tUp;
	double ZUp = point[3 - 1] - kz * tUp;

	if (YUp>YSide[2 - 1] && YUp<YSide[1 - 1] && ZUp>ZSide[2 - 1] && ZUp < ZSide[1 - 1])
	{
		;
	}
	else if (YUp > YSide[2 - 1] && YUp < YSide[1 - 1] && ZUp >= ZSide[1 - 1])
	{
		double Ztmp = kz * tUp + ZSide[1 - 1];
		centerPoint[3 - 1] = point[3 - 1] - 0.5*VoxSize + 0.5*(Ztmp - (point[3 - 1] - 0.5 * VoxSize));
	}
	else if (YUp>YSide[2 - 1] && YUp < YSide[1 - 1] && ZUp <= ZSide[2 - 1])
	{
		double Ztmp = kz * tUp + ZSide[2 - 1];
		centerPoint[3 - 1] = point[3 - 1] + 0.5 * VoxSize - 0.5 * (point[3 - 1] + 0.5 * VoxSize - Ztmp);
	}
	else if (ZUp>ZSide[2 - 1] && ZUp < ZSide[1 - 1] && YUp >= YSide[1 - 1])
	{
		double Ytmp = ky * tUp + YSide[1 - 1];
		centerPoint[2 - 1] = point[2 - 1] - 0.5*VoxSize + 0.5 * (Ytmp - (point[2 - 1] - 0.5 * VoxSize));
	}
	else if (ZUp > ZSide[2 - 1] && ZUp < ZSide[1 - 1] && YUp <= YSide[2 - 1])
	{
		double Ytmp = ky * tUp + YSide[2 - 1];
		centerPoint[2 - 1] = point[2 - 1] + 0.5 * VoxSize - 0.5*((point[2-1]+0.5*VoxSize) - Ytmp);
	}
	else if (ZUp >= ZSide[1 - 1] && YUp > YSide[1 - 1])
	{
		double Ytmp = ky * tUp + YSide[1 - 1];
		double Ztmp = kz * tUp + ZSide[1 - 1];
		centerPoint[2 - 1] = point[2 - 1] - 0.5*VoxSize + 0.5 * (Ytmp - (point[2 - 1] - 0.5*VoxSize));
		centerPoint[3 - 1] = point[3 - 1] - 0.5*VoxSize + 0.5 * (Ztmp - (point[3 - 1] - 0.5*VoxSize));
	}
	else if (ZUp <= ZSide[2 - 1] && YUp <= YSide[2 - 1])
	{
		double Ytmp = ky * tUp + YSide[2 - 1];
		double Ztmp = kz * tUp + ZSide[2 - 1];
		centerPoint[2 - 1] = point[2 - 1] + 0.5*VoxSize - 0.5 * (point[2 - 1] + 0.5*VoxSize - Ytmp);
		centerPoint[3 - 1] = point[3 - 1] + 0.5*VoxSize - 0.5 * (point[3 - 1] + 0.5*VoxSize - Ytmp);
	}
	else if (ZUp >= ZSide[1 - 1] && YUp <= YSide[2 - 1])
	{
		double Ytmp = ky * tUp + YSide[2 - 1];
		double Ztmp = kz * tUp + ZSide[1 - 1];
		centerPoint[2 - 1] = point[2 - 1] + 0.5 * VoxSize - 0.5*(point[2 - 1] + 0.5*VoxSize - Ytmp);
		centerPoint[3 - 1] = point[3 - 1] - 0.5 * VoxSize + 0.5*(Ztmp - (point[3 - 1] - 0.5*VoxSize));
	}
	else if (YUp >= YSide[1 - 1] && ZUp <= ZSide[2 - 1])
	{
		double Ytmp = ky * tUp + YSide[1 - 1];
		centerPoint[2 - 1] = point[2 - 1] - 0.5*VoxSize + 0.5 * (Ytmp - (point[2 - 1] - 0.5*VoxSize));
		double Ztmp = kz * tUp + ZSide[2 - 1];
		centerPoint[3 - 1] = point[3 - 1] + 0.5*VoxSize - 0.5 * (point[3 - 1] + 0.5 * VoxSize - Ztmp);
	}
	return centerPoint;
}

int main()
{
	//晶体数量
	const int CryNumY = 77, CryNumZ = 104;
	//晶体尺寸
	const int CrySize[3] = { 26, 4, 4 };
	//Y轴晶体中心
	double *CryCoorY = init_array_many_value(CryNumY, -(double)CryNumY*(double)CrySize[1] / 2 + (double)CrySize[1] / 2, (double)CrySize[1]);
	double *CryCoorZ = init_array_many_value(CryNumZ, -(double)CryNumZ*(double)CrySize[2] / 2 + (double)CrySize[2] / 2, (double)CrySize[2]);

	//showarray(CryCoorZ,104);
	//每个探测器的晶体数
	const int CryNumPerHead = CryNumY * CryNumZ;
	//两个探测器的距离
	const double Dis = 240;
	//LOR总数
	const int LORNum = CryNumY*CryNumY * CryNumZ*CryNumZ;
	const int VoxNumPerCry = 4;
	const int VoxNumX = 240, VoxNumY = 308, VoxNumZ = 416;
	const int VoxNumYZ = VoxNumY * VoxNumZ;
	const int VoxSize = 1;

	double *VoxCoorX = init_array_many_value(VoxNumX, -(double)VoxNumX*(double)VoxSize / 2 + (double)VoxSize / 2, (double)VoxSize);
	double *VoxCoorY = init_array_many_value(VoxNumY, -(double)VoxNumY*(double)VoxSize / 2 + (double)VoxSize / 2, (double)VoxSize);
	double *VoxCoorZ = init_array_many_value(VoxNumZ, -(double)VoxNumZ*(double)VoxSize / 2 + (double)VoxSize / 2, (double)VoxSize);




	double gap = 0.22;
	const int VoxNum = VoxNumYZ*VoxNumX;

	double nonzero_ratio[13] = { 0.0823, 0.1036, 0.1015, 0.0971, 0.0914, 0.0854, 0.0794, 0.0736, 0.0680, 0.0627, 0.0575, 0.0522, 0.0453 };
	double theta = 1;


	//cuda 已验证赋值
	double DeltaWeight[4] = { nonzero_ratio[0], sum_array(nonzero_ratio, 1, 2), sum_array(nonzero_ratio, 4, 6), sum_array(nonzero_ratio, 7, 12) };
	double DeepLen[4] = { 0, 2, 6, 14 };

	double OffAbandon = 0;
	double Start = 0;

	//定义及初始化
	double *norm = init_array_single_value(LORNum, 0);

	double u_LYSO = 0.087;
	double coeff = 1;

	int LORi = 1;
	int LORj = 1;

	//MATLAB从1开始，为了程序所有地方的数组引用都减一，这里从1开始（有的地方减有的不减不容易检查）
	for (int LORm = 1; LORm <= CryNumZ; LORm++)
	{
		for (int LORn = 1; LORn <= CryNumY; LORn++)
		{
			double *P = init_array_single_value(VoxNum, 0);
			double *tmp = init_array_single_value(VoxNum, 0);
			double *Solid = init_array_single_value(VoxNum, 0);
			
			for (int IndDoiUp = (Start + 1) ; IndDoiUp <= sizeof(DeepLen) / sizeof(double) - OffAbandon ; IndDoiUp++)
			{
				double weightup = DeltaWeight[IndDoiUp - 1];
				for (int IndDoiDown = (Start + 1); IndDoiDown <= sizeof(DeepLen) / sizeof(double) - OffAbandon; IndDoiUp++)
				{
					double weightdown = DeltaWeight[IndDoiDown];

					double OffsetUP = DeepLen[IndDoiUp-1];
					double OffsetDown = DeepLen[IndDoiDown - 1];

					double LORUp[3] = { Dis/2+DeepLen[IndDoiUp-1], CryCoorY[LORj-1], CryCoorZ[LORi-1] };
					double LORDown[3] = {-Dis/2-DeepLen[IndDoiDown-1], CryCoorY[LORn-1],CryCoorZ[LORm-1] };

					double kx = LORDown[1 - 1] - LORUp[1 - 1];
					double ky = LORDown[2 - 1] - LORUp[2 - 1];
					double kz = LORDown[3 - 1] - LORUp[3 - 1];

					double vecY[3] = {0,1,0};
					double vecZ[3] = {0,0,1};

					double lenLOR = sqrt(kx*kx+ky*ky+kz*kz);

					double angleY = acos(abs(ky) / lenLOR);
					double angleZ = acos(abs(kz) / lenLOR);

					double AttenLenUp = OffsetUP / (cos(atan(Dis / lenLOR)));
					double AttenLenDown = OffsetDown / (cos(atan(Dis / lenLOR)));

					double sliceEff = 1;

					if (ky == 0 && kz == 0)
					{
						int IndexY = ceil((LORUp[2 - 1] - (CryCoorY[1 - 1] - CrySize[2 - 1] / 2)) / CrySize[2 - 1]);
						int IndexZ = ceil((LORUp[3 - 1] - (CryCoorZ[1 - 1] - CrySize[3 - 1] / 2)) / CrySize[3 - 1]);

						int IndexVoxY = (IndexY - 1)*(double)VoxNumPerCry + 50 + 1;
						int IndexVoxZ = (IndexZ - 1)*(double)VoxNumPerCry + 50 + 1;

						for (int tmpXi = 1; tmpXi <= VoxNumX; tmpXi++)
						{
							double IndexTemp = (double)tmpXi + (IndexVoxY - 1)*VoxNumX + (IndexVoxZ - 1)*VoxNumY*VoxNumX;
							
							for (int Voxi = 1; Voxi <= VoxNumPerCry; Voxi++)
							{
								for (int Voxj = 1; Voxj <= VoxNumPerCry; Voxj++)
								{
									int Index = Index = IndexTemp + (Voxj - 1)*VoxNumX + (Voxi - 1)*VoxNumY*VoxNumX;
									double point[3] = {VoxCoorX[tmpXi-1],VoxCoorY[IndexVoxY+Voxj-1-1],VoxCoorZ[IndexVoxZ+Voxi-1-1]};

									double *centerPoint=findCen(point,LORUp,kx,ky,kz,CrySize,VoxSize,Dis,OffsetUP);
							

								}
							}
							
						}

					}

				}
			}
			free(P);
			free(tmp);
			free(Solid);
		}
	}


	system("pause");
	return 0;

}

