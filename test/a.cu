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

void showarray(float *a, int n)
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

__global__ void initsinglevalue(float *a, int n, float value)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int tid = blockDim.x * gridDim.x;
	while (i < n)
	{
		a[i] = value;
		i += tid;
	}
}



float* init_array_single_value(int num, float value)
{
	float *in;
	//大数组初始化，采用cuda
	in = (float*)malloc(num*sizeof(float));
	float *in_cuda;
	cudaMalloc((void**)&in_cuda, num*sizeof(float));
	initsinglevalue << <100, 100 >> >(in_cuda, num, value);
	cudaMemcpy(in, in_cuda, num*sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(in_cuda);
	return in;
}

__global__ void initmanyvalue(float *a, int n, float begin, float diff)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int tid = blockDim.x * gridDim.x;
	while (i < n)
	{
		a[i] = begin + i * diff;
		i += tid;
	}
}
float* init_array_many_value(int num, float begin, float diff)
{
	float *in;
	//等差数列初始化，采用cuda
	in = (float*)malloc(num*sizeof(float));
	float *in_cuda;
	cudaMalloc((void**)&in_cuda, num*sizeof(float));
	initmanyvalue << <1000, 1000 >> >(in_cuda, num, begin, diff);
	cudaMemcpy(in, in_cuda, num*sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(in_cuda);
	return in;
}

float sum_array(float *a, int n, int m)
{
	//矩阵从n加到m
	float out = 0;
	for (int i = n; i <= m; i++)
	{
		out += a[i];
	}
	return out;
}

float* findCen(float *point, float *LORUp, float kx, float ky, float kz, const int *CrySize, float VoxSize, float Distance, float OffsetUP)
{
	float *centerPoint;
	centerPoint = (float*)malloc(3 * sizeof(float));
	for (int i = 0; i < 3; i++)
	{
		centerPoint[i] = point[i];
	}

	float YSide[2] = { LORUp[2 - 1] + (float)CrySize[2 - 1] / 2, LORUp[2 - 1] - (float)CrySize[2 - 1] / 2 };
	float ZSide[2] = { LORUp[3 - 1] + (float)CrySize[3 - 1] / 2, LORUp[3 - 1] - (float)CrySize[3 - 1] / 2 };

	float tUp = (point[1 - 1] - Distance / 2 - OffsetUP) / kx;
	float YUp = point[2 - 1] - ky * tUp;
	float ZUp = point[3 - 1] - kz * tUp;

	if (YUp>YSide[2 - 1] && YUp<YSide[1 - 1] && ZUp>ZSide[2 - 1] && ZUp < ZSide[1 - 1])
	{
		;
	}
	else if (YUp > YSide[2 - 1] && YUp < YSide[1 - 1] && ZUp >= ZSide[1 - 1])
	{
		float Ztmp = kz * tUp + ZSide[1 - 1];
		centerPoint[3 - 1] = point[3 - 1] - 0.5*VoxSize + 0.5*(Ztmp - (point[3 - 1] - 0.5 * VoxSize));
	}
	else if (YUp>YSide[2 - 1] && YUp < YSide[1 - 1] && ZUp <= ZSide[2 - 1])
	{
		float Ztmp = kz * tUp + ZSide[2 - 1];
		centerPoint[3 - 1] = point[3 - 1] + 0.5 * VoxSize - 0.5 * (point[3 - 1] + 0.5 * VoxSize - Ztmp);
	}
	else if (ZUp>ZSide[2 - 1] && ZUp < ZSide[1 - 1] && YUp >= YSide[1 - 1])
	{
		float Ytmp = ky * tUp + YSide[1 - 1];
		centerPoint[2 - 1] = point[2 - 1] - 0.5*VoxSize + 0.5 * (Ytmp - (point[2 - 1] - 0.5 * VoxSize));
	}
	else if (ZUp > ZSide[2 - 1] && ZUp < ZSide[1 - 1] && YUp <= YSide[2 - 1])
	{
		float Ytmp = ky * tUp + YSide[2 - 1];
		centerPoint[2 - 1] = point[2 - 1] + 0.5 * VoxSize - 0.5*((point[2 - 1] + 0.5*VoxSize) - Ytmp);
	}
	else if (ZUp >= ZSide[1 - 1] && YUp > YSide[1 - 1])
	{
		float Ytmp = ky * tUp + YSide[1 - 1];
		float Ztmp = kz * tUp + ZSide[1 - 1];
		centerPoint[2 - 1] = point[2 - 1] - 0.5*VoxSize + 0.5 * (Ytmp - (point[2 - 1] - 0.5*VoxSize));
		centerPoint[3 - 1] = point[3 - 1] - 0.5*VoxSize + 0.5 * (Ztmp - (point[3 - 1] - 0.5*VoxSize));
	}
	else if (ZUp <= ZSide[2 - 1] && YUp <= YSide[2 - 1])
	{
		float Ytmp = ky * tUp + YSide[2 - 1];
		float Ztmp = kz * tUp + ZSide[2 - 1];
		centerPoint[2 - 1] = point[2 - 1] + 0.5*VoxSize - 0.5 * (point[2 - 1] + 0.5*VoxSize - Ytmp);
		centerPoint[3 - 1] = point[3 - 1] + 0.5*VoxSize - 0.5 * (point[3 - 1] + 0.5*VoxSize - Ztmp);
	}
	else if (ZUp >= ZSide[1 - 1] && YUp <= YSide[2 - 1])
	{
		float Ytmp = ky * tUp + YSide[2 - 1];
		float Ztmp = kz * tUp + ZSide[1 - 1];
		centerPoint[2 - 1] = point[2 - 1] + 0.5 * VoxSize - 0.5*(point[2 - 1] + 0.5*VoxSize - Ytmp);
		centerPoint[3 - 1] = point[3 - 1] - 0.5 * VoxSize + 0.5*(Ztmp - (point[3 - 1] - 0.5*VoxSize));
	}
	else if (YUp >= YSide[1 - 1] && ZUp <= ZSide[2 - 1])
	{
		float Ytmp = ky * tUp + YSide[1 - 1];
		centerPoint[2 - 1] = point[2 - 1] - 0.5*VoxSize + 0.5 * (Ytmp - (point[2 - 1] - 0.5*VoxSize));
		float Ztmp = kz * tUp + ZSide[2 - 1];
		centerPoint[3 - 1] = point[3 - 1] + 0.5*VoxSize - 0.5 * (point[3 - 1] + 0.5 * VoxSize - Ztmp);
	}
	return centerPoint;
}

float SolidAngle3D5(float *centerPoint, float *LORUp, float kx, float ky, float kz, const int *CrySize, float angleY, float angleZ, float Distance, float lenLOR, float OffsetUP)
{
	float tUp = (centerPoint[1 - 1] - Distance / 2 - OffsetUP) / kx;
	float YUp = centerPoint[2 - 1] - ky*tUp;
	float ZUp = centerPoint[3 - 1] - kz * tUp;

	float lenPtoUp = sqrt(pow((centerPoint[1 - 1] - Distance / 2 - OffsetUP), 2) + pow(centerPoint[2 - 1] - YUp, 2) + pow(centerPoint[3 - 1] - ZUp, 2));

	float lenYSide = abs(LORUp[2 - 1] + (float)CrySize[2 - 1] / 2 - YUp);
	float lenZSide = abs(LORUp[3 - 1] + (float)CrySize[3 - 1] / 2 - ZUp);

	float RY = lenYSide * sin(angleY);
	float RZ = lenZSide * sin(angleZ);

	float lenProY = CrySize[2 - 1] * sin(angleY);
	float lenProZ = CrySize[3 - 1] * sin(angleZ);

	RY = min(RY, lenProY - RY);
	RZ = min(RZ, lenProZ - RZ);

	float LY = lenPtoUp - (lenYSide)*cos(angleY);
	float LZ = lenPtoUp - (lenZSide)*cos(angleZ);

	float tmpY = (CrySize[2 - 1] - lenYSide) * cos(angleY);
	float tmpZ = (CrySize[3 - 1] - lenZSide) * cos(angleZ);

	float thetaY;
	float thetaZ;
	if (LY > 0 && LZ > 0)
	{
		float LY1 = lenLOR - lenPtoUp - tmpY;
		float LZ1 = lenLOR - lenPtoUp - tmpZ;
		if (LY1 > 0 && LZ1 > 0)
		{
			thetaY = atan(RY / LY) + atan(RY / LY1);
			thetaZ = atan(RZ / LZ) + atan(RZ / LZ1);

			float maxLY = max(LY, LY1);
			thetaY = min(atan(lenProY / maxLY), thetaY);
			float maxLZ = max(LZ, LZ1);
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

			float maxLY = max(LY, LY1);
			thetaY = min(atan(lenProY / maxLY), thetaY);
		}
		else if (LY1 <= 0 && LZ1 > 0)
		{
			thetaY = atan(lenProY / LY);
			thetaZ = atan(RZ / LZ) + atan(RZ / LZ1);
			float maxLZ = max(LZ, LZ1);
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
		float LY1 = lenLOR - lenPtoUp - tmpY;
		if (LY1 > 0)
		{
			thetaY = atan(RY / LY) + atan(RY / LY1);
			float maxLY = max(LY, LY1);
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
		float LZ1 = lenLOR - lenPtoUp - tmpZ;
		if (LZ1 > 0)
		{
			thetaZ = atan(RZ / LZ) + atan(RZ / LZ1);
			float maxLZ = max(LZ, LZ1);
			thetaZ = min(atan(lenProZ / maxLZ), thetaZ);
		}
		else
		{
			thetaZ = atan(lenProZ / LZ);
		}
		thetaY = atan(lenProY / (lenLOR - tmpY - lenPtoUp));
	}

	float theta1 = thetaY * thetaZ / (2 * 3.14159265358979323846);
	float sliceeff = 1;
	float theta = theta1 * sliceeff;
	return theta;

}



float * arraymult(float *array, float k, int n)
{
	float *out;
	out = (float*)malloc(n*sizeof(float));
	//数组乘法
	for (int i = 0; i < n; i++)
	{
		out[i] = k*array[i];
	}
	return out;
}

float * arrayadd(float *array, float k, int n)
{
	float *out;
	out = (float*)malloc(n*sizeof(float));
	//数组减法
	for (int i = 0; i < n; i++)
	{
		out[i] = array[i] + k;
	}
	return out;
}

int find_value_in_matrix(float *in, float min, float max, const int n)
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



//__global__ void find_interesect_index_and_sort2(float in1, float*in2, int n2, int *out)
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

__global__ void find_interesect_index_and_sort(float *in1, float*in2, int n1, int n2, float min1, float max1, float min2, float max2, int *out)
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

float mod(float a, float b)
{
	float c = a / b;
	return a - floor(c)*b;
}
__global__ void final_deal(float *tmp, float *P, float k, float n)
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

int findNonZeroValue(float *tmp, int *voxelIndex, float *weightValue, int n, int valueCount_SRM)
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
	float *CryCoorY = init_array_many_value(CryNumY, -(float)CryNumY*(float)CrySize[1] / 2 + (float)CrySize[1] / 2, (float)CrySize[1]);
	float *CryCoorZ = init_array_many_value(CryNumZ, -(float)CryNumZ*(float)CrySize[2] / 2 + (float)CrySize[2] / 2, (float)CrySize[2]);

	//showarray(CryCoorZ,104);
	//每个探测器的晶体数
	const int CryNumPerHead = CryNumY * CryNumZ;
	//两个探测器的距离
	const float Distance = 240;
	//X, Y, Z各个方向的体素数
	const int VoxNumX = 240;
	const int VoxNumY = 288;
	const int VoxNumZ = 420;
	//LOR总数
	const int LORNum = CryNumY*CryNumY * CryNumZ*CryNumZ;
	const int VoxNumPerCry = 4;
	const int VoxNumYZ = VoxNumY * VoxNumZ;
	float VoxSize = 1;

	float *VoxCoorX = init_array_many_value(VoxNumX, -(float)VoxNumX*VoxSize / 2 + VoxSize / 2, VoxSize);
	float *VoxCoorY = init_array_many_value(VoxNumY, -(float)VoxNumY*VoxSize / 2 + VoxSize / 2, VoxSize);
	float *VoxCoorZ = init_array_many_value(VoxNumZ, -(float)VoxNumZ*VoxSize / 2 + VoxSize / 2, VoxSize);

	float gap = 0.22;
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


	float nonzero_ratio[13] = { 0.082305172254121, 0.103600087979255, 0.101533524726801, 0.097068716428968, 0.091397666234488, 0.085401509538803, 0.079449018336729, 0.073608075569550, 0.068001620670495, 0.062668457121689, 0.057455524171143, 0.052245161140952, 0.045265465827005 };
	float theta = 1;


	//cuda 已验证赋值
	float DeltaWeight[4] = { nonzero_ratio[0], sum_array(nonzero_ratio, 1, 2), sum_array(nonzero_ratio, 4, 6), sum_array(nonzero_ratio, 7, 12) };

	float DeepLen[4] = { 0, 2, 6, 14 };

	float OffAbandon = 0;
	float Start = 0;

	//定义及初始化
	float *norm = init_array_single_value(LORNum, 0);

	float u_LYSO = 0.087;
	float coeff = 1;

	int LORi = 1;
	int LORj = 1;

	//响应线的序号
	int LOR_index = 0;

	//MATLAB从1开始，为了程序所有地方的数组引用都减一，这里从1开始（有的地方减有的不减不容易检查）
	//for (int LORm = 1; LORm <= CryNumZ; LORm++)
	for (int LORm = 9; LORm <= 9; LORm++)
	{
		//for (int LORn = 1; LORn <= CryNumY; LORn++)
		for (int LORn = 40; LORn <= 40; LORn++)
		{
			LOR_index++;

			printf("LORm=%d,LORn=%d\n", LORm, LORn);
			float *P = init_array_single_value(VoxNum, 0);
			float *tmp = init_array_single_value(VoxNum, 0);
			float *Solid = init_array_single_value(VoxNum, 0);

			for (int IndDoiUp = (Start + 1); IndDoiUp <= 4 - OffAbandon; IndDoiUp++)
			{

				float weightup = DeltaWeight[IndDoiUp - 1];
				for (int IndDoiDown = (Start + 1); IndDoiDown <= 4 - OffAbandon; IndDoiDown++)
				{
					//printf("InDoiDown=%d\n", IndDoiDown);
					float weightdown = DeltaWeight[IndDoiDown - 1];
					//printf("weightdown = %f\n", weightdown);
					//system("pause");

					float OffsetUP = DeepLen[IndDoiUp - 1];
					float OffsetDown = DeepLen[IndDoiDown - 1];

					float LORUp[3] = { Distance / 2 + DeepLen[IndDoiUp - 1], CryCoorY[LORj - 1], CryCoorZ[LORi - 1] };
					float LORDown[3] = { -Distance / 2 - DeepLen[IndDoiDown - 1], CryCoorY[LORn - 1], CryCoorZ[LORm - 1] };

					float LORUpLD[3] = { Distance / 2 + DeepLen[IndDoiUp - 1], CryCoorY[LORj - 1] - (float)CrySize[2 - 1] / 2 + VoxSize / 2, CryCoorZ[LORi - 1] - (float)CrySize[2 - 1] / 2 + VoxSize / 2 };
					float LORDownLD[3] = { -Distance / 2 - DeepLen[IndDoiDown - 1], CryCoorY[LORn - 1] - (float)CrySize[2 - 1] / 2 + VoxSize / 2, CryCoorZ[LORm - 1] - (float)CrySize[2 - 1] / 2 + VoxSize / 2 };

					float kx = LORDown[1 - 1] - LORUp[1 - 1];
					float ky = LORDown[2 - 1] - LORUp[2 - 1];
					float kz = LORDown[3 - 1] - LORUp[3 - 1];

					float vecY[3] = { 0, 1, 0 };
					float vecZ[3] = { 0, 0, 1 };

					float lenLOR = sqrt(kx*kx + ky*ky + kz*kz);

					float angleY = acos(abs(ky) / lenLOR);
					float angleZ = acos(abs(kz) / lenLOR);

					float AttenLenUp = OffsetUP / (cos(atan(Distance / lenLOR)));
					float AttenLenDown = OffsetDown / (cos(atan(Distance / lenLOR)));

					float sliceEff = 1;

					if (ky == 0 && kz == 0)
					{
						int IndexY = ceil((LORUp[2 - 1] - (CryCoorY[1 - 1] - (float)CrySize[2 - 1] / 2)) / (float)CrySize[2 - 1]);
						int IndexZ = ceil((LORUp[3 - 1] - (CryCoorZ[1 - 1] - (float)CrySize[3 - 1] / 2)) / (float)CrySize[3 - 1]);

						int IndexVoxY = (IndexY - 1)*(float)VoxNumPerCry + 50 + 1;
						int IndexVoxZ = (IndexZ - 1)*(float)VoxNumPerCry + 50 + 1;

						for (int tmpXi = 1; tmpXi <= VoxNumX; tmpXi++)
						{
							float IndexTemp = (float)tmpXi + (IndexVoxY - 1)*VoxNumX + (IndexVoxZ - 1)*VoxNumY*VoxNumX;
							
							for (int Voxi = 1; Voxi <= VoxNumPerCry; Voxi++)
							{
								for (int Voxj = 1; Voxj <= VoxNumPerCry; Voxj++)
								{
									int Index = IndexTemp + (Voxj - 1)*VoxNumX + (Voxi - 1)*VoxNumY*VoxNumX;
									float point[3] = { VoxCoorX[tmpXi - 1], VoxCoorY[IndexVoxY + Voxj - 1 - 1], VoxCoorZ[IndexVoxZ + Voxi - 1 - 1] };

									float *centerPoint = findCen(point, LORUp, kx, ky, kz, CrySize, VoxSize, Distance, OffsetUP);
									float theta = SolidAngle3D5(centerPoint, LORUp, kx, ky, kz, CrySize, angleY, angleZ, Distance, lenLOR, OffsetUP);
									free(centerPoint);
									P[Index - 1] = pow(VoxSize, 2) * theta*sliceEff;
								}
							}

						}


					}
					else if (ky != 0 || kz != 0)
					{
						float *X;
						X = (float*)malloc(sizeof(float)*VoxNumX);
						memcpy(X, VoxCoorX, VoxNumX*sizeof(float));

						float *t, *temp_x;
						temp_x = arrayadd(X, -LORUpLD[1 - 1], VoxNumX);
						t = arraymult(temp_x, 1 / kx, VoxNumX);
						free(temp_x);

						float *Y, *temp_y;
						temp_y = arraymult(t, ky, VoxNumX);
						Y = arrayadd(temp_y, LORUpLD[2 - 1], VoxNumX);
						free(temp_y);

						float *Z, *temp_z;
						temp_z = arraymult(t, kz, VoxNumX);
						Z = arrayadd(temp_z, LORUpLD[3 - 1], VoxNumX);
						free(temp_z);

						//这一步和matlab不一样，这一步后的YZ不会改变长度，而是输出len_Y、Z来表示他们的有效程度
						//注意YZ已经改变
						int len_Y = find_value_in_matrix(Y, VoxCoorY[1 - 1] - VoxSize / 2, VoxCoorY[VoxNumY - 1] + VoxSize / 2, VoxNumX);
						int len_Z = find_value_in_matrix(Z, VoxCoorZ[1 - 1] - VoxSize / 2, VoxCoorZ[VoxNumZ - 1] + VoxSize / 2, VoxNumX);


						if (len_Y > 0 && len_Z > 0)
						{
							float *Y_cuda, *Z_cuda;
							int *YZindex_cuda, *YZindex;
							cudaMalloc((void**)&Y_cuda, len_Y*sizeof(float));
							cudaMalloc((void**)&Z_cuda, len_Z*sizeof(float));
							cudaMalloc((void**)&YZindex_cuda, len_Z*sizeof(int));
							cudaMemcpy(Y_cuda, Y, len_Y*sizeof(float), cudaMemcpyHostToDevice);
							cudaMemcpy(Z_cuda, Z, len_Z*sizeof(float), cudaMemcpyHostToDevice);
							find_interesect_index_and_sort << <10, 24 >> >(Y_cuda, Z_cuda, len_Y, len_Z, \
								VoxCoorY[1 - 1] - VoxSize / 2, VoxCoorY[VoxNumY - 1] + VoxSize / 2, \
								VoxCoorZ[1 - 1] - VoxSize / 2, VoxCoorZ[VoxNumZ - 1] + VoxSize / 2, YZindex_cuda);
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
							float *VarInY, *VarInZ;
							int	*Index;

							IndexInX = (int *)malloc(len_X*sizeof(int));
							IndexInY = (int *)malloc(len_Y*sizeof(int));
							IndexInZ = (int *)malloc(len_Z*sizeof(int));

							VarInY = (float*)malloc(len_Y*sizeof(float));
							VarInZ = (float*)malloc(len_Z*sizeof(float));

							Index = (int*)malloc(len_Z*sizeof(int));
							for (int i = 0; i < len_X; i++)
							{
								float IndexInX_temp = (X[i] - (VoxCoorX[1 - 1] - VoxSize / 2)) / VoxSize;
								float IndexInY_temp = (Y[i] - (VoxCoorY[1 - 1] - VoxSize / 2)) / VoxSize;
								float IndexInZ_temp = (Z[i] - (VoxCoorZ[1 - 1] - VoxSize / 2)) / VoxSize;
								if (ceil(IndexInX_temp) - IndexInX_temp > 0.9999)
								{
									IndexInX_temp = ceil(IndexInX_temp) - 1;
								}
								else
								{
									IndexInX_temp = ceil(IndexInX_temp);
								}

								if (ceil(IndexInY_temp) - IndexInY_temp > 0.9999)
								{
									IndexInY_temp = ceil(IndexInY_temp) - 1;
								}
								else
								{
									IndexInY_temp = ceil(IndexInY_temp);
								}

								if (ceil(IndexInZ_temp) - IndexInZ_temp > 0.9999)
								{
									IndexInZ_temp = ceil(IndexInZ_temp) - 1;
								}
								else
								{
									IndexInZ_temp = ceil(IndexInZ_temp);
								}

								IndexInX[i] = IndexInX_temp;
								IndexInY[i] = IndexInY_temp;
								IndexInZ[i] = IndexInZ_temp;
								float modYV = mod(Y[i], VoxSize);//代替 mod(Y[i], VoxSize)
								float modZV = mod(Z[i], VoxSize);
								if (abs(modYV)<0.0001 || abs(modYV - 1)<0.0001)
								{
									modYV = 0;
								}
								if (abs(modZV)<0.0001 || abs(modZV - 1)<0.0001)
								{
									modZV = 0;
								}

								VarInY[i] = VoxSize - modYV;
								VarInZ[i] = VoxSize - modZV;

								//if (IndDoiUp == 3 && IndDoiDown == 3 && i == 88)
								//{
								//	printf("IndexInX[i]=%d\n", IndexInX[i]);
								//	printf("IndexInY[i]=%d\n", IndexInY[i]);
								//	printf("IndexInZ[i]=%d\n", IndexInZ[i]);
								//	printf("Y[i]=%f\n", Y[i]);
								//	printf("Z[i]=%f\n", Z[i]);
								//	printf("VoxCoorY[1 - 1]=%f\n", VoxCoorY[1 - 1]);
								//}

								Index[i] = IndexInX[i] + (IndexInY[i] - 1)*VoxNumX + (IndexInZ[i] - 1)*VoxNumX*VoxNumY;
							}
							

							for (int slicei = 1; slicei <= VoxNumX; slicei++)
							{
								//if (slicei == 39 && IndDoiUp == 4 && IndDoiDown == 4)
								//{
								//	printf("39\n");
								//	printf("VarInY[slicei - 1]=%.20f\n", VarInY[slicei - 1]);
								//	printf("VarInZ[slicei - 1]=%.20f\n", VarInZ[slicei - 1]);
								//	
								//	printf("Y[slicei-1]=%f", Y[slicei - 1]);
								//	printf("mod(Z[slicei - 1], VoxSize)=%f\n", mod(Y[slicei - 1], VoxSize));

								//}
								if (VarInY[slicei - 1] < 1 && VarInZ[slicei - 1] < 1 && abs(VarInY[slicei - 1] - 1)>0.00005 && abs(VarInZ[slicei - 1] - 1)>0.00005)
								{
									for (int tmpi = 1; tmpi <= VoxNumPerCry + 1; tmpi++)
									{
										for (int tmpj = 1; tmpj <= VoxNumPerCry + 1; tmpj++)
										{
											int IndexTmp = Index[slicei - 1] + (tmpj - 1)*VoxNumX + (tmpi - 1)*VoxNumX*VoxNumY;
											if (IndexTmp == 1541144)
											{
												printf("1 error\n");
												printf("slicei=%d\n", slicei);
												printf("IndDoiUp=%d\n", IndDoiUp);
												printf("IndDoiDown=%d\n", IndDoiDown);
												printf("VarInZ[slicei - 1]=%.20f\n", VarInZ[slicei - 1]);
												printf("VarInY[slicei - 1]=%.20f\n", VarInY[slicei - 1]);
												printf("Y[slicei-1]=%f\n", Y[slicei - 1]);
												printf("mod(Y[slicei - 1], VoxSize)=%f\n", mod(Y[slicei - 1], VoxSize));
												printf("Index[slicei - 1-1]=%f\n", Index[slicei - 1 - 1]);
												printf("Index[slicei - 1]=%f\n", Index[slicei - 1]);
												printf("Index[slicei - 1+1]=%f\n", Index[slicei - 1 + 1]);
											}
											float point[3] = { VoxCoorX[slicei - 1], VoxCoorY[IndexInY[slicei - 1] + tmpi - 1 - 1], VoxCoorZ[IndexInZ[slicei - 1] + tmpj - 1 - 1] };
											float *centerPoint = findCen(point, LORUp, kx, ky, kz, CrySize, VoxSize, Distance, OffsetUP);
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
								else if (VarInY[slicei - 1] < 1 && abs(VarInZ[slicei - 1]-1) < 0.00005 && abs(VarInY[slicei - 1] - 1)>0.00005)
								{
									for (int tmpi = 1; tmpi <= VoxNumPerCry; tmpi++)
									{
										for (int tmpj = 1; tmpj <= VoxNumPerCry + 1; tmpj++)
										{
											int IndexTmp = Index[slicei - 1] + (tmpj - 1)*VoxNumX + (tmpi - 1)*VoxNumX*VoxNumY;
											if (IndexTmp == 1541144)
											{
												printf("2 error\n");
												printf("VarInZ[slicei - 1]=%.20f\n", VarInZ[slicei - 1]);
												printf("VarInY[slicei - 1]=%.20f\n", VarInY[slicei - 1]);
												printf("IndDoiUp=%d\n", IndDoiUp);
												printf("IndDoiDown=%d\n", IndDoiDown);
												printf("slicei=%d\n", slicei);
												printf("VarInZ[slicei - 1]=%.20f\n", VarInZ[slicei - 1]);
												printf("VarInY[slicei - 1]=%.20f\n", VarInY[slicei - 1]);
												printf("Y[slicei-1]=%f\n", Y[slicei - 1]);
												printf("mod(Y[slicei - 1], VoxSize)=%f\n", mod(Y[slicei - 1], VoxSize));
											}
											float point[3] = { VoxCoorX[slicei - 1], VoxCoorY[IndexInY[slicei - 1] + tmpi - 1 - 1], VoxCoorZ[IndexInZ[slicei - 1] + tmpj - 1 - 1] };
											float *centerPoint = findCen(point, LORUp, kx, ky, kz, CrySize, VoxSize, Distance, OffsetUP);
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
								else if (abs(VarInY[slicei - 1] - 1) < 0.00005 && VarInZ[slicei - 1] < 1 && abs(VarInZ[slicei - 1]-1)>0.00005)
								{
									for (int tmpi = 1; tmpi <= VoxNumPerCry + 1; tmpi++)
									{
										for (int tmpj = 1; tmpj <= VoxNumPerCry; tmpj++)
										{

											int IndexTmp = Index[slicei - 1] + (tmpj - 1)*VoxNumX + (tmpi - 1)*VoxNumX*VoxNumY;
											if (IndexTmp == 1541144)
											{
												printf("3 error\n");
												printf("slicei=%d\n", slicei);
												printf("IndDoiUp=%d\n", IndDoiUp);
												printf("IndDoiDown=%d\n", IndDoiDown);
												printf("VarInZ[slicei - 1]=%.20f\n", VarInZ[slicei - 1]);
												printf("VarInY[slicei - 1]=%.20f\n", VarInY[slicei - 1]);
												printf("Y[slicei-1]=%f\n", Y[slicei - 1]);
												printf("mod(Y[slicei - 1], VoxSize)=%f\n", mod(Y[slicei - 1], VoxSize));
												printf("Index[slicei - 1-1]=%d\n", Index[slicei - 1 - 1]);
												printf("Index[slicei - 1]=%d\n", Index[slicei - 1]);
												printf("Index[slicei - 1+1]=%d\n", Index[slicei - 1 + 1]);
											}
											float point[3] = { VoxCoorX[slicei - 1], VoxCoorY[IndexInY[slicei - 1] + tmpi - 1 - 1], VoxCoorZ[IndexInZ[slicei - 1] + tmpj - 1 - 1] };
											float *centerPoint = findCen(point, LORUp, kx, ky, kz, CrySize, VoxSize, Distance, OffsetUP);
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
								else if (abs(VarInY[slicei - 1]-1) < 0.00001 && abs(VarInZ[slicei - 1]-1) <0.00001)
									for (int tmpi = 1; tmpi <= VoxNumPerCry; tmpi++)
									{
										for (int tmpj = 1; tmpj <= VoxNumPerCry; tmpj++)
										{
											int IndexTmp = Index[slicei - 1] + (tmpj - 1)*VoxNumX + (tmpi - 1)*VoxNumX*VoxNumY;
											if (IndexTmp == 1541144)
											{
												
												printf("4 error\n");
												printf("IndDoiUp=%d\n", IndDoiUp);
												printf("IndDoiDown=%d\n", IndDoiDown);
												printf("slicei=%d\n", slicei);
												printf("VarInZ[slicei - 1]=%.20f\n", VarInZ[slicei - 1]);
												printf("VarInY[slicei - 1]=%.20f\n", VarInY[slicei - 1]);
												printf("Y[slicei-1]=%f\n", Y[slicei - 1]);
												printf("mod(Y[slicei - 1], VoxSize)=%f\n", mod(Y[slicei - 1], VoxSize));
												printf("Index[slicei - 1-1]=%f\n", Index[slicei - 1-1]);
												printf("Index[slicei - 1]=%f\n", Index[slicei - 1]);
												printf("Index[slicei - 1+1]=%f\n", Index[slicei - 1+1]);

											}
											float point[3] = { VoxCoorX[slicei - 1], VoxCoorY[IndexInY[slicei - 1] + tmpi - 1 - 1], VoxCoorZ[IndexInZ[slicei - 1] + tmpj - 1 - 1] };
											float *centerPoint = findCen(point, LORUp, kx, ky, kz, CrySize, VoxSize, Distance, OffsetUP);
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
					float *tmp_cuda, *P_cuda;
					cudaMalloc((void**)&tmp_cuda, VoxNum*sizeof(float));
					cudaMalloc((void**)&P_cuda, VoxNum*sizeof(float));
					cudaMemcpy(tmp_cuda, tmp, VoxNum*sizeof(float), cudaMemcpyHostToDevice);
					cudaMemcpy(P_cuda, P, VoxNum*sizeof(float), cudaMemcpyHostToDevice);
					final_deal << <100, 100 >> >(tmp_cuda, P_cuda, coeff* weightup*weightdown, VoxNum);
					cudaMemcpy(tmp, tmp_cuda, VoxNum*sizeof(float), cudaMemcpyDeviceToHost);
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
	system("pause");
	return 0;

}

