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
	HANDLE_ERROR(cudaMalloc((void**)&in_cuda, num*sizeof(double)));
	initsinglevalue << <100, 100 >> >(in_cuda, num, value);
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

	double YSide[2] = { LORUp[2 - 1] + (double)CrySize[2 - 1] / 2, LORUp[2 - 1] - (double)CrySize[2 - 1] / 2 };
	double ZSide[2] = { LORUp[3 - 1] + (double)CrySize[3 - 1] / 2, LORUp[3 - 1] - (double)CrySize[3 - 1] / 2 };
	
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

double SolidAngle3D5(double *centerPoint, double *LORUp, double kx, double ky, double kz, const int *CrySize, double angleY, double angleZ, double Dis, double lenLOR, double OffsetUP)
{
	double tUp = (centerPoint[1 - 1] - Dis / 2 - OffsetUP) / kx;
	double YUp = centerPoint[2 - 1] - ky*tUp;
	double ZUp = centerPoint[3 - 1] - kz * tUp;

	double lenPtoUp = sqrt(pow((centerPoint[1-1]-Dis/2-OffsetUP),2) + pow(centerPoint[2-1]-YUp,2) + pow(centerPoint[3-1]-ZUp,2));

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
			thetaZ = atan(lenProY / (lenLOR - tmpY - lenPtoUp));
		}
	}

	double theta1 = thetaY * thetaZ / (2 * 3.1415926);
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

__global__ void find_interesect_index_and_sort(double *in1,double*in2,int n1,int n2,double min1,double max1,double min2,double max2, int *out)
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
__global__ void final_deal(double *tmp, double *P, double k,double n)
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

int  find_value_not_0_and_copy_it(double *tmp,double *voxelIndex,double *weightValue,int n)
{
	int index = 0;
	for (int i = 0; i < n; i++)
	{
		if (tmp[i] != 0)
		{
			voxelIndex[index] = i;
			weightValue[index] = tmp[i];
			index++;
		}
	}
	return index;
}

int main()
{
	//晶体数量
	const int CryNumY = 72, CryNumZ = 105;
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
	const int VoxNumX = 240, VoxNumY = 288, VoxNumZ = 420;
	const int VoxNumYZ = VoxNumY * VoxNumZ;
	const int VoxSize = 1;

	double *VoxCoorX = init_array_many_value(VoxNumX, -(double)VoxNumX*(double)VoxSize / 2 + (double)VoxSize / 2, (double)VoxSize);
	double *VoxCoorY = init_array_many_value(VoxNumY, -(double)VoxNumY*(double)VoxSize / 2 + (double)VoxSize / 2, (double)VoxSize);
	double *VoxCoorZ = init_array_many_value(VoxNumZ, -(double)VoxNumZ*(double)VoxSize / 2 + (double)VoxSize / 2, (double)VoxSize);




	double gap = 0.22;
	const int VoxNum = VoxNumYZ*VoxNumX;

	FILE *fod_nW, *fod_voxelIndex, *fod_weightValue;
	fod_nW = fopen("G:\\cuda\\number_72_105crystal_atten_Delta3_Method4.raw","a+");
	fod_voxelIndex = fopen("G:\\cuda\\indvoxel_72_105crystal_atten_Delta3_Method4.raw", "a+");
	fod_weightValue = fopen("G:\\cuda\\weight_72_105crystal_atten_Delta3_Method4.raw", "a+");

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
			printf("LORm=%d,LORn=%d\n",LORm,LORn);
			double *P = init_array_single_value(VoxNum, 0);
			double *tmp = init_array_single_value(VoxNum, 0);
			double *Solid = init_array_single_value(VoxNum, 0);
			
			for (int IndDoiUp = (Start + 1) ; IndDoiUp <= 4 - OffAbandon ; IndDoiUp++)
			{

				double weightup = DeltaWeight[IndDoiUp - 1];
				for (int IndDoiDown = (Start + 1); IndDoiDown <= 4 - OffAbandon; IndDoiDown++)
				{
					//printf("InDoiDown=%d\n", IndDoiDown);
					double weightdown = DeltaWeight[IndDoiDown-1];
					//printf("weightdown = %f\n", weightdown);
					//system("pause");

					double OffsetUP = DeepLen[IndDoiUp-1];
					double OffsetDown = DeepLen[IndDoiDown - 1];

					double LORUp[3] = { Dis/2+DeepLen[IndDoiUp-1], CryCoorY[LORj-1], CryCoorZ[LORi-1] };
					double LORDown[3] = {-Dis/2-DeepLen[IndDoiDown-1], CryCoorY[LORn-1],CryCoorZ[LORm-1] };

					double LORUpLD[3] = { Dis / 2 + DeepLen[IndDoiUp - 1], CryCoorY[LORj - 1] - (double)CrySize[2 - 1] / 2 + (double)VoxSize / 2, CryCoorZ[LORi - 1] - (double)CrySize[2 - 1] / 2 + (double)VoxSize / 2 };
					double LORDownLD[3] = { -Dis / 2 - DeepLen[IndDoiDown - 1], CryCoorY[LORn - 1] - (double)CrySize[2 - 1] / 2 + (double)VoxSize / 2, CryCoorZ[LORm - 1] - (double)CrySize[2 - 1] / 2 + (double)VoxSize / 2 };

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
									double point[3] = {VoxCoorX[tmpXi-1],VoxCoorY[IndexVoxY+Voxj-1-1],VoxCoorZ[IndexVoxZ+Voxi-1-1]};

									double *centerPoint=findCen(point,LORUp,kx,ky,kz,CrySize,VoxSize,Dis,OffsetUP);
									double theta = SolidAngle3D5(centerPoint, LORUp, kx, ky, kz, CrySize, angleY,angleZ,Dis,lenLOR, OffsetUP);
									P[Index-1] = pow(VoxSize,2) * theta*sliceEff;
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
						temp_x = arrayadd(X, -LORUpLD[1-1],VoxNumX);
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
						int len_Y = find_value_in_matrix(Y, VoxCoorY[1 - 1] - (double)VoxSize / 2, VoxCoorY[VoxNumY - 1] + (double)VoxSize / 2,VoxNumX);
						int len_Z = find_value_in_matrix(Z, VoxCoorZ[1 - 1] - (double)VoxSize / 2, VoxCoorZ[VoxNumZ - 1] + (double)VoxSize / 2, VoxNumX);
						
						if (len_Y > 0 && len_Z > 0)
						{
							double *Y_cuda, *Z_cuda;
							int *YZindex_cuda,*YZindex;
							cudaMalloc((void**)&Y_cuda, len_Y*sizeof(double));
							cudaMalloc((void**)&Z_cuda, len_Z*sizeof(double));
							cudaMalloc((void**)&YZindex_cuda, len_Z*sizeof(int));
							cudaMemcpy(Y_cuda, Y, len_Y*sizeof(double),cudaMemcpyHostToDevice);
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

							int len_X=0;
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
											double *centerPoint = findCen(point, LORUp, kx, ky, kz, CrySize, VoxSize, Dis, OffsetUP);
											theta = SolidAngle3D5(centerPoint, LORUp, kx, ky, kz, CrySize, angleY, angleZ, Dis, lenLOR, OffsetUP);
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
											double *centerPoint = findCen(point, LORUp, kx, ky, kz, CrySize, VoxSize, Dis, OffsetUP);
											theta = SolidAngle3D5(centerPoint, LORUp, kx, ky, kz, CrySize, angleY, angleZ, Dis, lenLOR, OffsetUP);
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
											double *centerPoint = findCen(point, LORUp, kx, ky, kz, CrySize, VoxSize, Dis, OffsetUP);
											theta = SolidAngle3D5(centerPoint, LORUp, kx, ky, kz, CrySize, angleY, angleZ, Dis, lenLOR, OffsetUP);
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
											double point[3] = { VoxCoorX[slicei-1],VoxCoorY[IndexInY[slicei-1] + tmpi - 1-1],VoxCoorZ[IndexInZ[slicei-1] + tmpj - 1-1] };
											double *centerPoint = findCen(point, LORUp, kx, ky, kz, CrySize, VoxSize, Dis, OffsetUP);
											theta = SolidAngle3D5(centerPoint, LORUp, kx, ky, kz, CrySize, angleY, angleZ, Dis, lenLOR, OffsetUP);
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
											P[IndexTmp-1] = pow(VoxSize, 2) * theta*sliceEff;
										}
									}
							}



							free(IndexInX);
							free(IndexInY);
							free(IndexInZ);
							free(VarInY);
							free(VarInZ);
							free(Index);

						}

						free(X);

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

			/////////////////
			double *voxelIndex, *weightValue;
			int *nW;
			nW = (int*)malloc(sizeof(int));
			voxelIndex = (double*)malloc(VoxNum*sizeof(double));
			weightValue = (double*)malloc(VoxNum*sizeof(double));
			*nW = find_value_not_0_and_copy_it(tmp,voxelIndex,weightValue,VoxNum);

			printf("nW = %d\n", *nW);
			fwrite(nW, sizeof(INT32), 1, fod_nW);
			fwrite(voxelIndex, sizeof(INT32), *nW, fod_voxelIndex);
			fwrite(weightValue, sizeof(float), *nW, fod_weightValue);

			free(P);
			free(tmp);
			free(Solid);
			free(voxelIndex);
			free(weightValue);
			free(nW);
		}
	}

	fclose(fod_nW);
	fclose(fod_voxelIndex);
	fclose(fod_weightValue);
	system("pause");
	return 0;

}

