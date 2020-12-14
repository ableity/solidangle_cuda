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
#include <cmath>

#include <stdlib.h>
#include <crtdbg.h>

using namespace std;

#ifndef max
#define max(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif



#ifdef _DEBUG
#define new new(_NORMAL_BLOCK, __FILE__, __LINE__)
#endif

void EnableMemLeakCheck()
{
	int tmpFlag = _CrtSetDbgFlag(_CRTDBG_REPORT_FLAG);
	tmpFlag |= _CRTDBG_LEAK_CHECK_DF;
	_CrtSetDbgFlag(tmpFlag);
}

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
		printf("]\n");
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
	cudaMalloc((void**)&in_cuda, num*sizeof(double));
	initsinglevalue << <100, 100 >> >(in_cuda, num, value);
	cudaMemcpy(in, in_cuda, num*sizeof(double), cudaMemcpyDeviceToHost);
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
	cudaMalloc((void**)&in_cuda, num*sizeof(double));
	initmanyvalue << <1000, 1000 >> >(in_cuda, num, begin, diff);
	cudaMemcpy(in, in_cuda, num*sizeof(double), cudaMemcpyDeviceToHost);
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

double* findCen(double *point, double *LORUp, double kx, double ky, double kz, const int *CrySize, double VoxSize, double Distance, double OffsetUP)
{
	double *centerPoint;
	centerPoint = (double*)malloc(3 * sizeof(double));
	for (int i = 0; i < 3; i++)
	{
		centerPoint[i] = point[i];
	}

	double YSide[2] = { LORUp[2 - 1] + (double)CrySize[2 - 1] / 2, LORUp[2 - 1] - (double)CrySize[2 - 1] / 2 };
	double ZSide[2] = { LORUp[3 - 1] + (double)CrySize[3 - 1] / 2, LORUp[3 - 1] - (double)CrySize[3 - 1] / 2 };

	double tUp = (point[1 - 1] - Distance / 2 - OffsetUP) / kx;
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
		centerPoint[2 - 1] = point[2 - 1] + 0.5 * VoxSize - 0.5*((point[2 - 1] + 0.5*VoxSize) - Ytmp);
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
		centerPoint[3 - 1] = point[3 - 1] + 0.5*VoxSize - 0.5 * (point[3 - 1] + 0.5*VoxSize - Ztmp);
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

double SolidAngle3D5(double *centerPoint, double *LORUp, double kx, double ky, double kz, const int *CrySize, double angleY, double angleZ, double Distance, double lenLOR, double OffsetUP)
{
	double tUp = (centerPoint[1 - 1] - Distance / 2 - OffsetUP) / kx;
	double YUp = centerPoint[2 - 1] - ky*tUp;
	double ZUp = centerPoint[3 - 1] - kz * tUp;

	double lenPtoUp = sqrt(pow((centerPoint[1 - 1] - Distance / 2 - OffsetUP), 2) + pow(centerPoint[2 - 1] - YUp, 2) + pow(centerPoint[3 - 1] - ZUp, 2));

	double lenYSide = abs(LORUp[2 - 1] + (double)CrySize[2 - 1] / 2 - YUp);
	double lenZSide = abs(LORUp[3 - 1] + (double)CrySize[3 - 1] / 2 - ZUp);

	double RY = lenYSide * sin(angleY);
	double RZ = lenZSide * sin(angleZ);

	double lenProY = CrySize[2 - 1] * sin(angleY);
	double lenProZ = CrySize[3 - 1] * sin(angleZ);

	RY = min(RY, lenProY - RY);
	RZ = min(RZ, lenProZ - RZ);

	double LY = lenPtoUp - (lenYSide)*cos(angleY);
	double LZ = lenPtoUp - (lenZSide)*cos(angleZ);

	double tmpY = (CrySize[2 - 1] - lenYSide) * cos(angleY);
	double tmpZ = (CrySize[3 - 1] - lenZSide) * cos(angleZ);

	double thetaY;
	double thetaZ;
	if (LY > 0 && LZ > 0)
	{
		double LY1 = lenLOR - lenPtoUp - tmpY;
		double LZ1 = lenLOR - lenPtoUp - tmpZ;
		if (LY1 > 0 && LZ1 > 0)
		{
			thetaY = atan(RY / LY) + atan(RY / LY1);
			thetaZ = atan(RZ / LZ) + atan(RZ / LZ1);

			double maxLY = max(LY, LY1);
			thetaY = min(atan(lenProY / maxLY), thetaY);
			double maxLZ = max(LZ, LZ1);
			thetaZ = min(atan(lenProZ / maxLZ), thetaZ);
		}
		else if (LY1 <= 0 && LZ1 <= 0)
		{
			thetaY = atan(lenProY / LY);
			thetaZ = atan(lenProZ / LZ);

		}
		else if (LY1 > 0 && LZ1 <= 0)
		{
			thetaY = atan(RY / LY) + atan(RY / LY1);
			thetaZ = atan(lenProZ / LZ);

			double maxLY = max(LY, LY1);
			double thetaY = min(atan(lenProY / maxLY), thetaY);
		}
		else if (LY1 <= 0 && LZ1 > 0)
		{
			thetaY = atan(lenProY / LY);
			thetaZ = atan(RZ / LZ) + atan(RZ / LZ1);
			double maxLZ = max(LZ, LZ1);
			thetaZ = min(atan(lenProZ / maxLZ), thetaZ);
		}
	}
	else if (LY <= 0 && LZ <= 0)
	{
		thetaY = atan(lenProY / (lenLOR - tmpY - lenPtoUp));
		thetaZ = atan(lenProZ / (lenLOR - tmpZ - lenPtoUp));
	}
	else if (LY > 0 && LZ <= 0)
	{
		double LY1 = lenLOR - lenPtoUp - tmpY;
		if (LY1 > 0)
		{
			thetaY = atan(RY / LY) + atan(RY / LY1);
			double maxLY = max(LY, LY1);
			thetaY = min(atan(lenProY / maxLY), thetaY);
		}
		else
		{
			thetaY = atan(lenProY / LY);
		}
		thetaZ = atan(lenProZ / (lenLOR - tmpZ - lenPtoUp));
	}
	else if (LY <= 0 && LZ > 0)
	{
		double LZ1 = lenLOR - lenPtoUp - tmpZ;
		if (LZ1 > 0)
		{
			thetaZ = atan(RZ / LZ) + atan(RZ / LZ1);
			double maxLZ = max(LZ, LZ1);
			thetaZ = min(atan(lenProZ / maxLZ), thetaZ);
		}
		else
		{
			thetaZ = atan(lenProZ / LZ);
		}
		thetaY = atan(lenProY / (lenLOR - tmpY - lenPtoUp));
	}

	double theta1 = thetaY * thetaZ / (2 * 3.14159265358979323846);
	double sliceeff = 1;
	double theta = theta1 * sliceeff;
	return theta;

}



double * arraymult(double *array, double k, int n)
{
	double *out;
	out = (double*)malloc(n*sizeof(double));
	//数组乘法
	for (int i = 0; i < n; i++)
	{
		out[i] = k*array[i];
	}
	return out;
}

double * arrayadd(double *array, double k, int n)
{
	double *out;
	out = (double*)malloc(n*sizeof(double));
	//数组减法
	for (int i = 0; i < n; i++)
	{
		out[i] = array[i] + k;
	}
	return out;
}

int find_value_in_matrix(double *in, double min, double max, const int n)
{
	//在原矩阵中返回介于min和max之间的值，其它在尾部置0
	//如{1,2,3,4,5} min=1.5,max=4.5,则{2,3,4...}返回3 
	int index = 0;
	for (int i = 0; i < n; i++)
	{
		if (in[i]>min && in[i] < max)
		{
			in[index] = in[i];
			index++;
		}
	}
	return index;

}



//__global__ void find_interesect_index_and_sort2(double in1, double*in2, int n2, int *out)
//{
//	int index = blockDim.x * blockIdx.x + threadIdx.x;
//	int tid = blockDim.x * gridDim.x;
//
//	while (index < n2)
//	{
//		out[index] = 0;
//		//printf("in1:%f\nin2[%d]:%f\n",in1,tid,in2[index]);
//		if (in1 == in2[index])
//		{
//			out[index] = 1 ;
//		}
//		index += tid;
//	}
//}

__global__ void find_interesect_index_and_sort(double *in1, double*in2, int n1, int n2, double min1, double max1, double min2, double max2, int *out)
{
	//找出对应的下标的交集并排序
	//其实不用排序，下标肯定是排好序的
	//本函数会二次启动核函数，注意内存
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	int index2 = blockDim.x * blockIdx.x + threadIdx.x;
	int tid = blockDim.x * gridDim.x;
	while (n2 > index2)
	{
		out[index2] = 0;
		index2 += tid;
	}
	while (n1 > index && n2> index)
	{
		if (in1[index] > min1 && in1[index] < max1 && in2[index] > min2 && in2[index] < max2)
		{
			out[index] = 1;
		}
		index += tid;
	}

}

double mod(double a, double b)
{
	double c = a / b;
	return a - floor(c)*b;
}
__global__ void final_deal(double *tmp, double *P, double k, double n)
{
	//tmp = P*k+tmp
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	int tid = blockDim.x * gridDim.x;

	while (index < n)
	{
		tmp[index] = P[index] * k + tmp[index];
		index += tid;
	}
}

int findNonZeroValue(double *tmp, int *voxelIndex, float *weightValue, int n, int valueCount_SRM)
{
	int index = 0;
	for (int i = 0; i < n; i++)
	{
		if (tmp[i] != 0)
		{
			voxelIndex[valueCount_SRM + index] = i + 1;
			weightValue[valueCount_SRM + index] = tmp[i];
			index++;
		}
	}
	return index;
}

int main()
{
	//晶体数量

	EnableMemLeakCheck();

	//_CrtSetBreakAlloc(927662);

	const int CryNumY = 72;
	const int CryNumZ = 105;
	//晶体尺寸
	const int CrySize[3] = { 26, 4, 4 };
	//Y轴晶体中心
	double *CryCoorY = init_array_many_value(CryNumY, -(double)CryNumY*(double)CrySize[1] / 2 + (double)CrySize[1] / 2, (double)CrySize[1]);
	double *CryCoorZ = init_array_many_value(CryNumZ, -(double)CryNumZ*(double)CrySize[2] / 2 + (double)CrySize[2] / 2, (double)CrySize[2]);

	//showarray(CryCoorZ,104);
	//每个探测器的晶体数
	const int CryNumPerHead = CryNumY * CryNumZ;
	//两个探测器的距离
	const double Distance = 240;
	//X, Y, Z各个方向的体素数
	const int VoxNumX = 240;
	const int VoxNumY = 288;
	const int VoxNumZ = 420;
	//LOR总数
	const int LORNum = CryNumY*CryNumY * CryNumZ*CryNumZ;
	const int VoxNumPerCry = 4;
	const int VoxNumYZ = VoxNumY * VoxNumZ;
	const int VoxSize = 1;

	double *VoxCoorX = init_array_many_value(VoxNumX, -(double)VoxNumX*(double)VoxSize / 2 + (double)VoxSize / 2, (double)VoxSize);
	double *VoxCoorY = init_array_many_value(VoxNumY, -(double)VoxNumY*(double)VoxSize / 2 + (double)VoxSize / 2, (double)VoxSize);
	double *VoxCoorZ = init_array_many_value(VoxNumZ, -(double)VoxNumZ*(double)VoxSize / 2 + (double)VoxSize / 2, (double)VoxSize);

	double gap = 0.22;
	//体素总的个数
	const int VoxNum = VoxNumYZ*VoxNumX;

	//提前开辟空间储存系统矩阵的weght, indvoxel,number
	int MaxSize = 400000000;//平板乳腺PET系统矩阵最大有效值不超过这个数
	//int MaxSize = 30000000;//小动物PET系统矩阵最大有效值不超过这个数

	int *number = new int[CryNumY*CryNumZ];
	int *indvoxel = new int[MaxSize];
	float *weight = new float[MaxSize];

	//记录系统矩阵总有效值的个数
	int valueCount_SRM = 0;


	double nonzero_ratio[13] = { 0.082305172254121, 0.103600087979255, 0.101533524726801, 0.097068716428968, 0.091397666234488, 0.085401509538803, 0.079449018336729, 0.073608075569550, 0.068001620670495, 0.062668457121689, 0.057455524171143, 0.052245161140952, 0.045265465827005 };
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

	//响应线的序号
	int LOR_index = 0;

	//MATLAB从1开始，为了程序所有地方的数组引用都减一，这里从1开始（有的地方减有的不减不容易检查）
	for (int LORm = 1; LORm <= CryNumZ; LORm++)
	//for (int LORm = 1; LORm <= 1; LORm++)
	{
		for (int LORn = 1; LORn <= CryNumY; LORn++)
		//for (int LORn = 1; LORn <= 10; LORn++)
		{
			LOR_index++;

			printf("LORm=%d,LORn=%d\n", LORm, LORn);
			double *P = init_array_single_value(VoxNum, 0);
			double *tmp = init_array_single_value(VoxNum, 0);
			double *Solid = init_array_single_value(VoxNum, 0);

			for (int IndDoiUp = (Start + 1); IndDoiUp <= 4 - OffAbandon; IndDoiUp++)
			{

				double weightup = DeltaWeight[IndDoiUp - 1];
				for (int IndDoiDown = (Start + 1); IndDoiDown <= 4 - OffAbandon; IndDoiDown++)
				{
					//printf("InDoiDown=%d\n", IndDoiDown);
					double weightdown = DeltaWeight[IndDoiDown - 1];
					//printf("weightdown = %f\n", weightdown);
					//system("pause");

					double OffsetUP = DeepLen[IndDoiUp - 1];
					double OffsetDown = DeepLen[IndDoiDown - 1];

					double LORUp[3] = { Distance / 2 + DeepLen[IndDoiUp - 1], CryCoorY[LORj - 1], CryCoorZ[LORi - 1] };
					double LORDown[3] = { -Distance / 2 - DeepLen[IndDoiDown - 1], CryCoorY[LORn - 1], CryCoorZ[LORm - 1] };

					double LORUpLD[3] = { Distance / 2 + DeepLen[IndDoiUp - 1], CryCoorY[LORj - 1] - (double)CrySize[2 - 1] / 2 + (double)VoxSize / 2, CryCoorZ[LORi - 1] - (double)CrySize[2 - 1] / 2 + (double)VoxSize / 2 };
					double LORDownLD[3] = { -Distance / 2 - DeepLen[IndDoiDown - 1], CryCoorY[LORn - 1] - (double)CrySize[2 - 1] / 2 + (double)VoxSize / 2, CryCoorZ[LORm - 1] - (double)CrySize[2 - 1] / 2 + (double)VoxSize / 2 };

					double kx = LORDown[1 - 1] - LORUp[1 - 1];
					double ky = LORDown[2 - 1] - LORUp[2 - 1];
					double kz = LORDown[3 - 1] - LORUp[3 - 1];

					double vecY[3] = { 0, 1, 0 };
					double vecZ[3] = { 0, 0, 1 };

					double lenLOR = sqrt(kx*kx + ky*ky + kz*kz);

					double angleY = acos(abs(ky) / lenLOR);
					double angleZ = acos(abs(kz) / lenLOR);

					double AttenLenUp = OffsetUP / (cos(atan(Distance / lenLOR)));
					double AttenLenDown = OffsetDown / (cos(atan(Distance / lenLOR)));

					double sliceEff = 1;

					if (ky == 0 && kz == 0)
					{
						int IndexY = ceil((LORUp[2 - 1] - (CryCoorY[1 - 1] - (double)CrySize[2 - 1] / 2)) / (double)CrySize[2 - 1]);
						int IndexZ = ceil((LORUp[3 - 1] - (CryCoorZ[1 - 1] - (double)CrySize[3 - 1] / 2)) / (double)CrySize[3 - 1]);

						int IndexVoxY = (IndexY - 1)*(double)VoxNumPerCry + 50 + 1;
						int IndexVoxZ = (IndexZ - 1)*(double)VoxNumPerCry + 50 + 1;

						for (int tmpXi = 1; tmpXi <= VoxNumX; tmpXi++)
						{
							double IndexTemp = (double)tmpXi + (IndexVoxY - 1)*VoxNumX + (IndexVoxZ - 1)*VoxNumY*VoxNumX;

							for (int Voxi = 1; Voxi <= VoxNumPerCry; Voxi++)
							{
								for (int Voxj = 1; Voxj <= VoxNumPerCry; Voxj++)
								{
									int Index = IndexTemp + (Voxj - 1)*VoxNumX + (Voxi - 1)*VoxNumY*VoxNumX;
									double point[3] = { VoxCoorX[tmpXi - 1], VoxCoorY[IndexVoxY + Voxj - 1 - 1], VoxCoorZ[IndexVoxZ + Voxi - 1 - 1] };

									double *centerPoint = findCen(point, LORUp, kx, ky, kz, CrySize, VoxSize, Distance, OffsetUP);
									double theta = SolidAngle3D5(centerPoint, LORUp, kx, ky, kz, CrySize, angleY, angleZ, Distance, lenLOR, OffsetUP);
									free(centerPoint);
									P[Index - 1] = pow(VoxSize, 2) * theta*sliceEff;
								}
							}

						}


					}
					else if (ky != 0 || kz != 0)
					{
						double *X;
						X = (double*)malloc(sizeof(double)*VoxNumX);
						memcpy(X, VoxCoorX, VoxNumX*sizeof(double));

						double *t, *temp_x;
						temp_x = arrayadd(X, -LORUpLD[1 - 1], VoxNumX);
						t = arraymult(temp_x, 1 / kx, VoxNumX);
						free(temp_x);

						double *Y, *temp_y;
						temp_y = arraymult(t, ky, VoxNumX);
						Y = arrayadd(temp_y, LORUpLD[2 - 1], VoxNumX);
						free(temp_y);

						double *Z, *temp_z;
						temp_z = arraymult(t, kz, VoxNumX);
						Z = arrayadd(temp_z, LORUpLD[3 - 1], VoxNumX);
						free(temp_z);

						//这一步和matlab不一样，这一步后的YZ不会改变长度，而是输出len_Y、Z来表示他们的有效程度
						//注意YZ已经改变
						int len_Y = find_value_in_matrix(Y, VoxCoorY[1 - 1] - (double)VoxSize / 2, VoxCoorY[VoxNumY - 1] + (double)VoxSize / 2, VoxNumX);
						int len_Z = find_value_in_matrix(Z, VoxCoorZ[1 - 1] - (double)VoxSize / 2, VoxCoorZ[VoxNumZ - 1] + (double)VoxSize / 2, VoxNumX);


						if (len_Y > 0 && len_Z > 0)
						{
							double *Y_cuda, *Z_cuda;
							int *YZindex_cuda, *YZindex;
							cudaMalloc((void**)&Y_cuda, len_Y*sizeof(double));
							cudaMalloc((void**)&Z_cuda, len_Z*sizeof(double));
							cudaMalloc((void**)&YZindex_cuda, len_Z*sizeof(int));
							cudaMemcpy(Y_cuda, Y, len_Y*sizeof(double), cudaMemcpyHostToDevice);
							cudaMemcpy(Z_cuda, Z, len_Z*sizeof(double), cudaMemcpyHostToDevice);
							find_interesect_index_and_sort << <10, 24 >> >(Y_cuda, Z_cuda, len_Y, len_Z, \
								VoxCoorY[1 - 1] - (double)VoxSize / 2, VoxCoorY[VoxNumY - 1] + (double)VoxSize / 2, \
								VoxCoorZ[1 - 1] - (double)VoxSize / 2, VoxCoorZ[VoxNumZ - 1] + (double)VoxSize / 2, YZindex_cuda);
							YZindex = (int *)malloc(len_Z*sizeof(int));
							//YZindex和matlab版不同，matlab版的是存的index，这里是index位置为1
							cudaMemcpy(YZindex, YZindex_cuda, len_Z * sizeof(int), cudaMemcpyDeviceToHost);

							cudaFree(Y_cuda);
							cudaFree(Z_cuda);
							cudaFree(YZindex_cuda);

							int len_X = 0;
							//下面循环是为了完成
							//X = X(YZIndex);
							//Y = Y(YZIndex);
							//Z = Z(YZIndex);
							//可以优化成一步
							for (int i = 0; i < len_Z; i++)
							{
								if (YZindex[i] == 1)
								{
									X[len_X] = X[i];
									Y[len_X] = Y[i];
									Z[len_X] = Z[i];
									len_X += 1;
								}

							}
							len_Y = len_X;
							len_Z = len_X;
							int *IndexInX, *IndexInY, *IndexInZ;
							double *VarInY, *VarInZ;
							double	*Index;

							IndexInX = (int *)malloc(len_X*sizeof(int));
							IndexInY = (int *)malloc(len_Y*sizeof(int));
							IndexInZ = (int *)malloc(len_Z*sizeof(int));

							VarInY = (double*)malloc(len_Y*sizeof(double));
							VarInZ = (double*)malloc(len_Z*sizeof(double));

							Index = (double*)malloc(len_Z*sizeof(double));
							for (int i = 0; i < len_X; i++)
							{

								IndexInX[i] = ceil((X[i] - (VoxCoorX[1 - 1] - (double)VoxSize / 2)) / (double)VoxSize);
								IndexInY[i] = ceil((Y[i] - (VoxCoorY[1 - 1] - (double)VoxSize / 2)) / (double)VoxSize);
								IndexInZ[i] = ceil((Z[i] - (VoxCoorZ[1 - 1] - (double)VoxSize / 2)) / (double)VoxSize);
								VarInY[i] = (double)VoxSize - mod(Y[i], (double)VoxSize);
								VarInZ[i] = (double)VoxSize - mod(Z[i], (double)VoxSize);

								Index[i] = IndexInX[i] + (IndexInY[i] - 1)*(double)VoxNumX + (IndexInZ[i] - 1)*(double)VoxNumX*(double)VoxNumY;
							}

							for (int slicei = 1; slicei <= VoxNumX; slicei++)
							{
								if (VarInY[slicei - 1] < 1 && VarInZ[slicei - 1] < 1)
								{
									for (int tmpi = 1; tmpi <= VoxNumPerCry + 1; tmpi++)
									{
										for (int tmpj = 1; tmpj <= VoxNumPerCry + 1; tmpj++)
										{
											int IndexTmp = Index[slicei - 1] + (tmpj - 1)*VoxNumX + (tmpi - 1)*VoxNumX*VoxNumY;
											double point[3] = { VoxCoorX[slicei - 1], VoxCoorY[IndexInY[slicei - 1] + tmpi - 1 - 1], VoxCoorZ[IndexInZ[slicei - 1] + tmpj - 1 - 1] };
											double *centerPoint = findCen(point, LORUp, kx, ky, kz, CrySize, VoxSize, Distance, OffsetUP);
											theta = SolidAngle3D5(centerPoint, LORUp, kx, ky, kz, CrySize, angleY, angleZ, Distance, lenLOR, OffsetUP);
											free(centerPoint);
											if ((tmpi != 1 || tmpi != VoxNumPerCry + 1) && (tmpj != 1 || tmpj != VoxNumPerCry + 1))
											{
												P[IndexTmp - 1] = theta;
											}
											else if (tmpi == 1 && (tmpj != 1 || tmpj != VoxNumPerCry + 1))
											{
												P[IndexTmp - 1] = theta;
											}
											else if (tmpi == VoxNumPerCry + 1 && (tmpj != 1 || tmpj != VoxNumPerCry + 1))
											{
												P[IndexTmp - 1] = theta;
											}
											else if ((tmpi != 1 || tmpi != VoxNumPerCry + 1) && tmpj == 1)
											{
												P[IndexTmp - 1] = theta;
											}
											else if ((tmpi != 1 || tmpi != VoxNumPerCry + 1) && tmpj == VoxNumPerCry + 1)
											{
												P[IndexTmp - 1] = theta;
											}
											else if (tmpi == 1 && tmpj == 1)
											{
												P[IndexTmp - 1] = theta;
											}
											else if (tmpi == 1 && tmpj == VoxNumPerCry + 1)
											{
												P[IndexTmp - 1] = theta;
											}
											else if (tmpi == 5 && tmpj == 1)
											{
												P[IndexTmp - 1] = theta;
											}
											else if (tmpi == 5 && tmpj == VoxNumPerCry + 1)
											{
												P[IndexTmp - 1] = theta;
											}
										}
									}
								}
								else if (VarInY[slicei - 1] < 1 && VarInZ[slicei - 1] == 1)
								{
									for (int tmpi = 1; tmpi <= VoxNumPerCry; tmpi++)
									{
										for (int tmpj = 1; tmpj <= VoxNumPerCry + 1; tmpj++)
										{
											int IndexTmp = Index[slicei - 1] + (tmpj - 1)*VoxNumX + (tmpi - 1)*VoxNumX*VoxNumY;
											double point[3] = { VoxCoorX[slicei - 1], VoxCoorY[IndexInY[slicei - 1] + tmpi - 1 - 1], VoxCoorZ[IndexInZ[slicei - 1] + tmpj - 1 - 1] };
											double *centerPoint = findCen(point, LORUp, kx, ky, kz, CrySize, VoxSize, Distance, OffsetUP);
											theta = SolidAngle3D5(centerPoint, LORUp, kx, ky, kz, CrySize, angleY, angleZ, Distance, lenLOR, OffsetUP);
											free(centerPoint);
											if (tmpj != 1 || tmpj != VoxNumPerCry + 1)
											{
												P[IndexTmp - 1] = theta;
											}
											else if (tmpj == 1)
											{
												P[IndexTmp - 1] = theta;
											}
											else if (tmpj == VoxNumPerCry + 1)
											{
												P[IndexTmp - 1] = theta;
											}
										}
									}
								}
								else if (VarInY[slicei - 1] == 1 && VarInZ[slicei - 1] < 1)
								{
									for (int tmpi = 1; tmpi <= VoxNumPerCry + 1; tmpi++)
									{
										for (int tmpj = 1; tmpj <= VoxNumPerCry + 1; tmpj++)
										{
											int IndexTmp = Index[slicei - 1] + (tmpj - 1)*VoxNumX + (tmpi - 1)*VoxNumX*VoxNumY;
											double point[3] = { VoxCoorX[slicei - 1], VoxCoorY[IndexInY[slicei - 1] + tmpi - 1 - 1], VoxCoorZ[IndexInZ[slicei - 1] + tmpj - 1 - 1] };
											double *centerPoint = findCen(point, LORUp, kx, ky, kz, CrySize, VoxSize, Distance, OffsetUP);
											theta = SolidAngle3D5(centerPoint, LORUp, kx, ky, kz, CrySize, angleY, angleZ, Distance, lenLOR, OffsetUP);
											free(centerPoint);
											if (tmpi != 1 || tmpj != VoxNumPerCry + 1)
											{
												P[IndexTmp - 1] = theta;
											}
											else if (tmpi == 1)
											{
												P[IndexTmp - 1] = theta;
											}
											else if (tmpi == VoxNumPerCry + 1)
											{
												P[IndexTmp - 1] = theta;
											}
										}
									}
								}
								else if (VarInY[slicei - 1] == 1 && VarInZ[slicei - 1] == 1)
									for (int tmpi = 1; tmpi <= VoxNumPerCry; tmpi++)
									{
										for (int tmpj = 1; tmpj <= VoxNumPerCry; tmpj++)
										{
											int IndexTmp = Index[slicei - 1] + (tmpj - 1)*VoxNumX + (tmpi - 1)*VoxNumX*VoxNumY;
											double point[3] = { VoxCoorX[slicei - 1], VoxCoorY[IndexInY[slicei - 1] + tmpi - 1 - 1], VoxCoorZ[IndexInZ[slicei - 1] + tmpj - 1 - 1] };
											double *centerPoint = findCen(point, LORUp, kx, ky, kz, CrySize, VoxSize, Distance, OffsetUP);
											theta = SolidAngle3D5(centerPoint, LORUp, kx, ky, kz, CrySize, angleY, angleZ, Distance, lenLOR, OffsetUP);
											free(centerPoint);
											//printf("IndexTmp=%d\n", IndexTmp);
											//printf("slicei=%d\n", slicei);
											//printf("tmpi=%d\n", tmpi);
											//printf("tmpj=%d\n", tmpj);
											//showarray(point, 3);
											//showarray(centerPoint, 3);
											//printf("LORUp=%f\n", LORUp);
											//printf("kx=%f\n", kx);
											//printf("ky=%f\n", ky);
											//printf("kz=%f\n", kz);
											//printf("angleY=%f\n", angleY);
											//printf("angleZ=%f\n", angleZ);
											//printf("lenLOR=%f\n", lenLOR);
											//printf("OffsetUP=%f\n", OffsetUP);
											//printf("theta = %f\n", theta);
											//system("pause");
											P[IndexTmp - 1] = pow(VoxSize, 2) * theta*sliceEff;
										}
									}
							}


							free(YZindex);
							free(IndexInX);
							free(IndexInY);
							free(IndexInZ);
							free(VarInY);
							free(VarInZ);
							free(Index);

						}

						free(X);
						free(Z);
						free(Y);
						free(t);
					}
					//tmp = P*coeff* weightup*weightdown + tmp;
					double *tmp_cuda, *P_cuda;
					cudaMalloc((void**)&tmp_cuda, VoxNum*sizeof(double));
					cudaMalloc((void**)&P_cuda, VoxNum*sizeof(double));
					cudaMemcpy(tmp_cuda, tmp, VoxNum*sizeof(double), cudaMemcpyHostToDevice);
					cudaMemcpy(P_cuda, P, VoxNum*sizeof(double), cudaMemcpyHostToDevice);
					final_deal << <100, 100 >> >(tmp_cuda, P_cuda, coeff* weightup*weightdown, VoxNum);
					cudaMemcpy(tmp, tmp_cuda, VoxNum*sizeof(double), cudaMemcpyDeviceToHost);
					//printf("tmp[3708001]=%f\n",tmp[3708001]);
					cudaFree(tmp_cuda);
					cudaFree(P_cuda);
				}
			}

			//******************************************************************************//
			//统计当前LOR 对应的有效数据个数
			int nW = 0;
			nW = findNonZeroValue(tmp, indvoxel, weight, VoxNum, valueCount_SRM);
			printf("nW = %d\n", nW);
			valueCount_SRM += nW;
			number[LOR_index - 1] = nW;

			//******************************************************************************//

			free(P);
			free(tmp);
			free(Solid);

		}
	}


	cout << "系统矩阵有效数据总个数：" << valueCount_SRM << endl;
	//保存系统矩阵
	FILE *fid;
	fopen_s(&fid, "G:\\cuda\\number_72_105crystal_atten_Delta3_Method4.raw", "wb");
	fwrite(number, sizeof(int), CryNumY*CryNumZ, fid);
	fclose(fid);
	fopen_s(&fid, "G:\\cuda\\indvoxel_72_105crystal_atten_Delta3_Method4.raw", "wb");
	fwrite(indvoxel, sizeof(int), valueCount_SRM, fid);
	fclose(fid);
	fopen_s(&fid, "G:\\cuda\\weight_72_105crystal_atten_Delta3_Method4.raw", "wb");
	fwrite(weight, sizeof(float), valueCount_SRM, fid);
	fclose(fid);

	free(number);
	free(indvoxel);
	free(weight);
	free(norm);
	free(CryCoorY);
	free(CryCoorZ);
	free(VoxCoorX);
	free(VoxCoorY);
	free(VoxCoorZ);
	return 0;

}

