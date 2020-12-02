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
using namespace std;

void showarray(double *a,int n)
{
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
		for (int i = n-4; i < n; i++)
		{
			if (i != n-4)
				printf(",");
			printf("%.4f", a[i]);
		}
		printf("]\n");
	}
}

__global__ void initvalue(double *a, int n, double value)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int tid = blockDim.x * gridDim.x;
	while (i < n)
	{
		a[i] = value;
		i += tid;
	}
}


double* init_big_array(int num, double value)
{
	double *in;
	//大数组初始化，采用cuda
	in = (double*)malloc(num*sizeof(double));
	double *in_cuda;
	HANDLE_ERROR(cudaMalloc((void**)&in_cuda, num*sizeof(double)));
	initvalue << <1000, 1000 >> >(in_cuda, num, value);
	HANDLE_ERROR(cudaMemcpy(in, in_cuda, num*sizeof(double), cudaMemcpyDeviceToHost));
	return in;
}

double sum_array(double *a, int n, int m)
{
	double out = 0;
	for (int i = n; i <= m; i++)
	{
		out += a[i];
	}
	return out;
}

int main()
{
	//晶体数量
	const int CryNumY = 77, CryNumZ = 104;
	//晶体尺寸
	const int CrySize[3] = {26,4,4}; 
	//Y轴晶体中心
	double CryCoorY[CryNumY];
	for (int i = 0; i < CryNumY; i++)
	{
		//赋值，cuda 已验证
		CryCoorY[i] = -(double)CryNumY*(double)CrySize[1] / 2 + (double)CrySize[1]/2+(double)i*(double)CrySize[1];
	}
	double CryCoorZ[CryNumZ];
	for (int i = 0; i < CryNumZ; i++)
	{
		//赋值，cuda 已验证
		CryCoorZ[i] = -(double)CryNumZ*(double)CrySize[2] / 2 + (double)CrySize[2]/2+(double)i*(double)CrySize[2];
	}

	//showarray(CryCoorZ,104);
	//每个探测器的晶体数
	const int CryNumPerHead = CryNumY * CryNumZ;
	//两个探测器的距离
	const double Dis = 240;
	//LOR总数
	const int LORNum = CryNumY*CryNumY * CryNumZ*CryNumZ;
	const int VoxNumX = 240, VoxNumY = 308, VoxNumZ = 416;
	const int VoxNumYZ = VoxNumY * VoxNumZ;
	const int VoxSize = 1;
	double VoxCoorX[VoxNumX],VoxCoorY[VoxNumY],VoxCoorZ[VoxNumZ];
	for (int i = 0; i < VoxNumX; i++)
	{
		//cuda 已验证
		VoxCoorX[i] = -(double)VoxNumX*(double)VoxSize / 2 + (double)VoxSize / 2 + (double)i*(double)VoxSize;
	}
	for (int i = 0; i < VoxNumY; i++)
	{
		VoxCoorY[i] = -(double)VoxNumY*(double)VoxSize / 2 + (double)VoxSize / 2 + (double)i*(double)VoxSize;
	}
	for (int i = 0; i < VoxNumZ; i++)
	{
		VoxCoorZ[i] = -(double)VoxNumZ*(double)VoxSize / 2 + (double)VoxSize / 2 + (double)i*(double)VoxSize;
	}

	double gap = 0.22;
	const int VoxNum = VoxNumYZ*VoxNumX;

	double nonzero_ratio[13]={0.0823, 0.1036, 0.1015, 0.0971, 0.0914, 0.0854, 0.0794, 0.0736, 0.0680, 0.0627, 0.0575, 0.0522, 0.0453 };
	double theta = 1;


	//cuda 已验证赋值
	double DeltaWeight[4] = { nonzero_ratio[0], sum_array(nonzero_ratio, 1, 2), sum_array(nonzero_ratio, 4, 6), sum_array(nonzero_ratio, 7, 12) };
	double DeepLen[4] = { 0, 2, 6, 14 };

	double offAbandon = 0;
	double Start = 0;

	//定义及初始化
	double *norm = init_big_array(LORNum, 0);

	double u_LYSO = 0.087;
	double coeff = 1;

	int LORi = 1;
	int LORj = 1;


	//MATLAB从1开始，为了程序所有地方的数组引用都减一，这里从1开始（有的地方减有的不减不容易检查）
	for (int LORm = 1; LORm <= CryNumZ; LORm++)
	{
		for (int LORn = 1; LORn <= CryNumY; LORn++)
		{

		}
	}


	system("pause");
	return 0;

}


