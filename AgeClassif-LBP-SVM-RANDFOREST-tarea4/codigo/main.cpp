#define _DEBUG
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>

using namespace cv;
using namespace cv::ml;
using namespace std;
Mat transformadaLBP(Mat imagen)
{
	cvtColor(imagen, imagen, CV_BGR2GRAY);
	int LUT[]={1,2,3,4,5,0,6,7,8,0,0,0,9,0,10,11,12,0,0,0,0,0,0,0,13,0,0,0,14,0,15,16,17,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,18,0,0,0,0,0,0,0,19,0,0,0,20,0,21,22,23,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,24,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,25,0,0,0,0,0,0,0,26,0,0,0,27,0,28,29,30,31,0,32,0,0,0,33,0,0,0,0,0,0,0,34,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,35,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,36,37,38,0,39,0,0,0,40,0,0,0,0,0,0,0,41,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,42,43,44,0,45,0,0,0,46,0,0,0,0,0,0,0,47,48,49,0,50,0,0,0,51,52,53,0,54,55,56,57,58};
	Mat LBP= Mat::zeros(imagen.rows-2, imagen.cols-2, CV_32FC1);
	for (int f=1; f<imagen.rows-1; f++)
	{
		for (int c=1; c<imagen.cols-1; c++)
		{
			int pixel=0;
			int binario[8]={0,0,0,0,0,0,0,0};
			if(imagen.at<uchar>(f-1,c-1) >= imagen.at<uchar>(f,c))
				binario[0]=1;
			if(imagen.at<uchar>(f-1,c) >= imagen.at<uchar>(f,c))
				binario[1]=1;
			if(imagen.at<uchar>(f-1,c+1) >= imagen.at<uchar>(f,c))
				binario[2]=1;
			if(imagen.at<uchar>(f,c+1) >= imagen.at<uchar>(f,c))
				binario[3]=1;
			if(imagen.at<uchar>(f+1,c+1) >= imagen.at<uchar>(f,c))
				binario[4]=1;
			if(imagen.at<uchar>(f+1,c) >= imagen.at<uchar>(f,c))
				binario[5]=1;
			if(imagen.at<uchar>(f+1,c-1) >= imagen.at<uchar>(f,c))
				binario[6]=1;
			if(imagen.at<uchar>(f,c-1) >= imagen.at<uchar>(f,c))
			{	
				binario[7]=1;
			}
			pixel=binario[0]*pow(2,7)+binario[1]*pow(2,6)+binario[2]*pow(2,5)+binario[3]*pow(2,4)+binario[4]*pow(2,3)+binario[5]*pow(2,2)+binario[6]*pow(2,1)+binario[7]*pow(2,0);	
			LBP.at<float>(f-1,c-1)=LUT[pixel];
		}	
	}
	Mat output;
	LBP.convertTo(output, CV_8UC1);
	return output;//la imagen resultante solo tiene numeros del 0 al 58, es decir, tiene 59 casillas o posibles valores
}

Mat nhistogramaLBP(Mat imagenLBP, int n)
{
	Mat histograma_concatenated;	
	int filas_ventana = imagenLBP.rows/n;
	int columnas_ventana = imagenLBP.cols/n;
	for (int f = 0; f<n; f++)//para recorresr verticalmente las ventanas
	{		
		for(int c= 0; c<n; c++)//para recorrer horizontalmente las ventanas
		{
			Mat ventana = Mat::zeros(filas_ventana, columnas_ventana, CV_8UC1);

			for(int fv = 0; fv < filas_ventana; fv++)
			{
				for(int cv = 0; cv < columnas_ventana; cv++)
				{
					ventana.at<uchar>(fv,cv) = imagenLBP.at<uchar>((f*filas_ventana) + fv, (c*columnas_ventana) + cv);
				} 
			}
			cv::Mat histograma_ventana;
			int histSize = 59;
    		float range[] = {0, 59};
    		const float* histRange = {range};	
			cv::calcHist(&ventana, 1, 0, cv::Mat(), histograma_ventana, 1, &histSize, &histRange, true, false);

			if(f==0 && c==0)
			{
				histograma_concatenated = histograma_ventana.t();
			}	else	{
				cv::hconcat(histograma_concatenated, histograma_ventana.t(), histograma_concatenated);		
			}
		} 
	}
	return histograma_concatenated;
}
void makeTrainTest_db1(int n, Mat &train_features, Mat &test_features, Mat &train_labels, Mat &test_labels, string path_build)
{
	vector<String> path_imagenes_db1;
	string pre_path_db1 = "db_tarea_4/db1/*.jpg";
	string path_db1 = path_build + pre_path_db1;

	Mat features, labels;
	glob(path_db1,path_imagenes_db1,true);
	
	for(int i=0; i<path_imagenes_db1.size();i++)
	{
		int label = 0;
		//cout<<"i = "<<i<<endl;
		//cout << path_imagenes_db1[i] << endl;
		Mat src = imread(path_imagenes_db1[i]); 
		if (src.empty()) 
		{
			continue;
		}

		Mat imagenLBP=transformadaLBP(src);
		Mat histograma_actual=nhistogramaLBP(imagenLBP, n);
		
		if(i < 200)
		{
			label=0;
		}
		if(i >= 200)
		{
			label=1;
		}  

		cv::Mat label_actual(1, 1, CV_32SC1, cv::Scalar(label));

		if (i == 0) 
		{
			features = histograma_actual;
			labels = label_actual;
		} else {
			cv::vconcat(features, histograma_actual, features);
			cv::vconcat(labels, label_actual, labels);
		}
		
	}	

	Mat train_features_clase0 = Mat::zeros(140, 59*n*n, CV_32FC1);
	Mat train_features_clase1 = Mat::zeros(140, 59*n*n, CV_32FC1);
	cv::Mat train_labels_clase0(140, 1, CV_32SC1, cv::Scalar(0));
	cv::Mat train_labels_clase1(140, 1, CV_32SC1, cv::Scalar(0));
	
	Mat test_features_clase0 = Mat::zeros(60, 59*n*n, CV_32FC1);
	Mat test_features_clase1 = Mat::zeros(60, 59*n*n, CV_32FC1);
	cv::Mat test_labels_clase0(60, 1, CV_32SC1, cv::Scalar(0));
	cv::Mat test_labels_clase1(60, 1, CV_32SC1, cv::Scalar(0));
	

	for(int i=0;i<140;i++)//para train clase 0
	{
		for (int f=0;f<features.cols;f++){
			train_features_clase0.at<float>(i,f)=features.at<float>(i,f);
		}	

		train_labels_clase0.at<int>(i,0) = labels.at<int>(i,0);
	}
	
	for(int i=140;i<200;i++)//para test clase 0
	{
		for (int f=0;f<features.cols;f++){
			test_features_clase0.at<float>(i-140,f) = features.at<float>(i,f);
		}		
		test_labels_clase0.at<int>(i-140,0) = labels.at<int>(i,0);
	}

	for(int i=200;i<340;i++)//para train clase 1
	{
		for (int f=0;f<features.cols;f++){
			train_features_clase1.at<float>(i-200,f)=features.at<float>(i,f);
		}	
		train_labels_clase1.at<int>(i-200,0) = labels.at<int>(i,0);
	}
	
	for(int i=340;i<400;i++)//para test clase 1
	{
		for (int f=0;f<features.cols;f++){
			test_features_clase1.at<float>(i-340,f) = features.at<float>(i,f);
		}		
		test_labels_clase1.at<int>(i-340,0) = labels.at<int>(i,0);
	}

	cv::vconcat(train_features_clase0, train_features_clase1, train_features);
	cv::vconcat(train_labels_clase0, train_labels_clase1, train_labels);
	
	cv::vconcat(test_features_clase0, test_features_clase1, test_features);
	cv::vconcat(test_labels_clase0, test_labels_clase1, test_labels);
}
		
void makeTrainTest_db2(int n, Mat &train_features, Mat &test_features, Mat &train_labels, Mat &test_labels, string path_build)
{
	vector<String> path_imagenes_db2;
	string pre_path_db2 = "db_tarea_4/db2/*.jpg";
	string path_db2 = path_build + pre_path_db2;

	Mat features, labels;	
	glob(path_db2,path_imagenes_db2,true);
	for(int i=0; i<path_imagenes_db2.size();i++)
	{
		int label = 0;
		Mat src = imread(path_imagenes_db2[i]); 
		if (src.empty()) 
		{
			continue;
		}
		
		Mat imagenLBP=transformadaLBP(src);
		Mat histograma_actual=nhistogramaLBP(imagenLBP,n);
		if(i < 200)
		{
			label=0;
		}
		if(i >= 200 && i < 400)
		{
			label=2;
		}  
		if(i >= 400 && i < 600)
		{
			label=1;
		}  

		cv::Mat label_actual(1, 1, CV_32SC1, cv::Scalar(label));

		if (i == 0) 
		{
			features = histograma_actual;
			labels = label_actual;
		} else {
			cv::vconcat(features, histograma_actual, features);
			cv::vconcat(labels, label_actual, labels);
		}
	}

	Mat train_features_clase0 = Mat::zeros(140, 59*n*n, CV_32FC1);
	Mat train_features_clase1 = Mat::zeros(140, 59*n*n, CV_32FC1);
	Mat train_features_clase2 = Mat::zeros(140, 59*n*n, CV_32FC1);
	cv::Mat train_labels_clase0(140, 1, CV_32SC1, cv::Scalar(0));
	cv::Mat train_labels_clase1(140, 1, CV_32SC1, cv::Scalar(1));
	cv::Mat train_labels_clase2(140, 1, CV_32SC1, cv::Scalar(2));
	
	Mat test_features_clase0 = Mat::zeros(60, 59*n*n, CV_32FC1);
	Mat test_features_clase1 = Mat::zeros(60, 59*n*n, CV_32FC1);
	Mat test_features_clase2 = Mat::zeros(60, 59*n*n, CV_32FC1);

	cv::Mat test_labels_clase0(60, 1, CV_32SC1, cv::Scalar(0));
	cv::Mat test_labels_clase1(60, 1, CV_32SC1, cv::Scalar(1));
	cv::Mat test_labels_clase2(60, 1, CV_32SC1, cv::Scalar(2));
	

	for(int i=0;i<140;i++)//para train clase 0
	{
		for (int f=0;f<features.cols;f++){
			train_features_clase0.at<float>(i,f)=features.at<float>(i,f);
		}	

		train_labels_clase0.at<int>(i,0) = labels.at<int>(i,0);
	}
	
	for(int i=140;i<200;i++)//para test clase 0
	{
		for (int f=0;f<features.cols;f++){
			test_features_clase0.at<float>(i-140,f) = features.at<float>(i,f);
		}		
		test_labels_clase0.at<int>(i-140,0) = labels.at<int>(i,0);
	}


	for(int i=200;i<340;i++)//para train clase 2
	{
		for (int f=0;f<features.cols;f++){
			train_features_clase2.at<float>(i-200,f)=features.at<float>(i,f);
		}	
		train_labels_clase2.at<int>(i-200,0) = labels.at<int>(i,0);
	}
	
	for(int i=340;i<400;i++)//para test clase 2
	{
		for (int f=0;f<features.cols;f++){
			test_features_clase2.at<float>(i-340,f) = features.at<float>(i,f);
		}		
		test_labels_clase2.at<int>(i-340,0) = labels.at<int>(i,0);
	}

	for(int i=400;i<540;i++)//para train clase 1
	{
		for (int f=0;f<features.cols;f++){
			train_features_clase1.at<float>(i-400,f)=features.at<float>(i,f);
		}	
		train_labels_clase1.at<int>(i-400,0) = labels.at<int>(i,0);
	}

	for(int i=540;i<600;i++)//para test clase 1
	{
		for (int f=0;f<features.cols;f++){
			test_features_clase1.at<float>(i-540,f) = features.at<float>(i,f);
		}		
		test_labels_clase1.at<int>(i-540,0) = labels.at<int>(i,0);
	}
	
	cv::vconcat(train_features_clase0, train_features_clase1, train_features);
	cv::vconcat(train_features, train_features_clase2, train_features);
	
	cv::vconcat(train_labels_clase0, train_labels_clase1, train_labels);
	cv::vconcat(train_labels, train_labels_clase2, train_labels);

	cv::vconcat(test_features_clase0, test_features_clase1, test_features);
	cv::vconcat(test_features, test_features_clase2, test_features);

	cv::vconcat(test_labels_clase0, test_labels_clase1, test_labels);
	cv::vconcat(test_labels, test_labels_clase2, test_labels);
}

void caller(int tipoproblema, string path_build)
{
	cout<<"-------------------------"<<endl;
	if(tipoproblema == 1)
	{
		int n = 4;
		Mat train_features = Mat::zeros(280, 59*n*n, CV_32FC1);
		Mat test_features = Mat::zeros(120, 59*n*n, CV_32FC1);
		Mat train_labels(280, 1, CV_32SC1, cv::Scalar(0));
		Mat test_labels(120, 1, CV_32SC1, cv::Scalar(1));
		
		makeTrainTest_db1(n, train_features, test_features, train_labels, test_labels, path_build);

		Ptr<SVM> svm = SVM::create();
    	svm->setType(SVM::C_SVC);
    	svm->setKernel(SVM::LINEAR);
    	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
    	
		svm->train(train_features, ROW_SAMPLE, train_labels);
			
		cv::Mat predicted_labels(120, 1, CV_32SC1, cv::Scalar(1));
		int sum_vp_y_vn=0;
		for(int i=0; i<test_features.rows; i++)
		{
			int prediction = svm->predict(test_features.row(i));
			predicted_labels.at<int>(i,0)=prediction;
			if(test_labels.at<int>(i,0)==predicted_labels.at<int>(i,0))
			{
				sum_vp_y_vn = sum_vp_y_vn+1;
			}
		}
		cout<<"SUMA DE LA DIAGONAL = " << sum_vp_y_vn<<endl;
		cout<<"accuracy SVM 2 clases (n=4)  : " << (sum_vp_y_vn/120.0)*100<<endl;
	}

	if(tipoproblema == 2)
	{
		int n = 4;
		Mat train_features = Mat::zeros(280, 59*n*n, CV_32FC1);
		Mat test_features = Mat::zeros(120, 59*n*n, CV_32FC1);
		Mat train_labels(280, 1, CV_32SC1, cv::Scalar(0));
		Mat test_labels(120, 1, CV_32SC1, cv::Scalar(1));
		
		makeTrainTest_db1(n, train_features, test_features, train_labels, test_labels, path_build);
		
		//create RandomForest classifier	
		auto rf = cv::ml::RTrees::create();		
		//train RandomForest
		rf->train(train_features,ROW_SAMPLE, train_labels);

		//Predict with RandomForest
		cv::Mat predicted_labels(120, 1, CV_32SC1, cv::Scalar(1));
		
		int sum_vp_y_vn=0;
		for(int i=0; i<test_features.rows; i++)
		{
			int prediction = rf->predict(test_features.row(i));
			predicted_labels.at<int>(i,0)=prediction;
		
			//Evaluation of predition
			if(test_labels.at<int>(i,0)==predicted_labels.at<int>(i,0))
			{
				sum_vp_y_vn=sum_vp_y_vn+1;
			}
		}
		//Show accuracy
		cout<<"SUMA DE LA DIAGONAL = " << sum_vp_y_vn<<endl;
		cout<<"accuracy RandomForest 2 clases (n=4)  : " << (sum_vp_y_vn/120.0)*100<<endl;
	}
	
	if(tipoproblema == 3)
	{
		int n = 4;
		Mat train_features = Mat::zeros(420, 59*n*n, CV_32FC1);
		Mat test_features = Mat::zeros(180, 59*n*n, CV_32FC1);
		Mat train_labels(420, 1, CV_32SC1, cv::Scalar(0));
		Mat test_labels(180, 1, CV_32SC1, cv::Scalar(0));
		
		makeTrainTest_db2(n, train_features, test_features, train_labels, test_labels, path_build);

		Ptr<SVM> svm = SVM::create();
    	svm->setType(SVM::C_SVC);
    	svm->setKernel(SVM::LINEAR);
    	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));

		svm->train(train_features, ROW_SAMPLE, train_labels);
		//svm->trainAuto(train_features, train_labels);
        	
		cv::Mat predicted_labels(180, 1, CV_32SC1, cv::Scalar(1));
		int sum_vp_y_vn=0;
		for(int i=0; i<test_features.rows; i++)
		{
			int prediction = svm->predict(test_features.row(i));
			predicted_labels.at<int>(i,0)=prediction;
			
			if(test_labels.at<int>(i,0)==predicted_labels.at<int>(i,0))
			{
				sum_vp_y_vn=sum_vp_y_vn+1;
			}
		}
		cout<<"SUMA DE LA DIAGONAL = " << sum_vp_y_vn<<endl;
		cout<<"accuracy SVM 3 clases (n=4)  : " << (sum_vp_y_vn/180.0)*100<<endl;
	}

	if(tipoproblema == 4)
	{
		int n = 4;
		Mat train_features = Mat::zeros(420, 59*n*n, CV_32FC1);
		Mat test_features = Mat::zeros(180, 59*n*n, CV_32FC1);
		Mat train_labels(420, 1, CV_32SC1, cv::Scalar(0));
		Mat test_labels(180, 1, CV_32SC1, cv::Scalar(0));
		
		makeTrainTest_db2(n, train_features, test_features, train_labels, test_labels, path_build);

		//create RandomForest classifier	
		auto rf = cv::ml::RTrees::create();		
		//train RandomForest
		rf->train(train_features,ROW_SAMPLE, train_labels);

		//Predict with RandomForest
		cv::Mat predicted_labels(180, 1, CV_32SC1, cv::Scalar(1));
		
		int sum_vp_y_vn=0;
		for(int i=0; i<test_features.rows; i++)
		{
			int prediction = rf->predict(test_features.row(i));
			predicted_labels.at<int>(i,0)=prediction;
			//Evaluation of predition
			if(test_labels.at<int>(i,0)==predicted_labels.at<int>(i,0))
			{
				sum_vp_y_vn = sum_vp_y_vn+1;
			}
		}
		//Show accuracy
		cout<<"SUMA DE LA DIAGONAL = " << sum_vp_y_vn<<endl;
		cout<<"accuracy RandomForest 3 clases (n=4)  : " << (sum_vp_y_vn/180.0)*100<<endl;
	}
	
	if(tipoproblema == 5)
	{
		int n = 2;
		Mat train_features = Mat::zeros(280, 59*n*n, CV_32FC1);
		Mat test_features = Mat::zeros(120, 59*n*n, CV_32FC1);
		Mat train_labels(280, 1, CV_32SC1, cv::Scalar(0));
		Mat test_labels(120, 1, CV_32SC1, cv::Scalar(1));
		
		makeTrainTest_db1(n, train_features, test_features, train_labels, test_labels, path_build);

		Ptr<SVM> svm = SVM::create();
    	svm->setType(SVM::C_SVC);
    	svm->setKernel(SVM::LINEAR);
    	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
    	
		svm->train(train_features, ROW_SAMPLE, train_labels);
			
		cv::Mat predicted_labels(120, 1, CV_32SC1, cv::Scalar(1));
		int sum_vp_y_vn=0;
		for(int i=0; i<test_features.rows; i++)
		{
			int prediction = svm->predict(test_features.row(i));
			predicted_labels.at<int>(i,0)=prediction;
			if(test_labels.at<int>(i,0)==predicted_labels.at<int>(i,0))
			{
				sum_vp_y_vn = sum_vp_y_vn+1;
			}
		}
		cout<<"SUMA DE LA DIAGONAL = " << sum_vp_y_vn<<endl;
		cout<<"accuracy SVM 2 clases (n=2)  : " << (sum_vp_y_vn/120.0)*100<<endl;
	}

	if(tipoproblema == 6)
	{
		int n = 2;
		Mat train_features = Mat::zeros(280, 59*n*n, CV_32FC1);
		Mat test_features = Mat::zeros(120, 59*n*n, CV_32FC1);
		Mat train_labels(280, 1, CV_32SC1, cv::Scalar(0));
		Mat test_labels(120, 1, CV_32SC1, cv::Scalar(1));
		
		makeTrainTest_db1(n, train_features, test_features, train_labels, test_labels, path_build);
		
		//create RandomForest classifier	
		auto rf = cv::ml::RTrees::create();		
		//train RandomForest
		rf->train(train_features,ROW_SAMPLE, train_labels);

		//Predict with RandomForest
		cv::Mat predicted_labels(120, 1, CV_32SC1, cv::Scalar(1));
		
		int sum_vp_y_vn=0;
		for(int i=0; i<test_features.rows; i++)
		{
			int prediction = rf->predict(test_features.row(i));
			predicted_labels.at<int>(i,0)=prediction;
			//Evaluation of predition
			if(test_labels.at<int>(i,0)==predicted_labels.at<int>(i,0))
			{
				sum_vp_y_vn=sum_vp_y_vn+1;
			}
		}
		//Show accuracy
		cout<<"SUMA DE LA DIAGONAL = " << sum_vp_y_vn<<endl;
		cout<<"accuracy RandomForest 2 clases (n=2)  : " << (sum_vp_y_vn/120.0)*100<<endl;
	}
	
	if(tipoproblema == 7)
	{
		int n = 2;
		Mat train_features = Mat::zeros(420, 59*n*n, CV_32FC1);
		Mat test_features = Mat::zeros(180, 59*n*n, CV_32FC1);
		Mat train_labels(420, 1, CV_32SC1, cv::Scalar(0));
		Mat test_labels(180, 1, CV_32SC1, cv::Scalar(0));
		
		makeTrainTest_db2(n, train_features, test_features, train_labels, test_labels, path_build);

		Ptr<SVM> svm = SVM::create();
    	svm->setType(SVM::C_SVC);
    	svm->setKernel(SVM::LINEAR);
    	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));

		svm->train(train_features, ROW_SAMPLE, train_labels);
        	
		cv::Mat predicted_labels(180, 1, CV_32SC1, cv::Scalar(1));
		int sum_vp_y_vn=0;
		for(int i=0; i<test_features.rows; i++)
		{
			int prediction = svm->predict(test_features.row(i));
			predicted_labels.at<int>(i,0)=prediction;
			
			if(test_labels.at<int>(i,0)==predicted_labels.at<int>(i,0))
			{
				sum_vp_y_vn=sum_vp_y_vn+1;
			}
		}
		cout<<"SUMA DE LA DIAGONAL = " << sum_vp_y_vn<<endl;
		cout<<"accuracy SVM 3 clases (n=2)  : " << (sum_vp_y_vn/180.0)*100<<endl;
	}

	if(tipoproblema == 8)
	{
		int n = 2;
		Mat train_features = Mat::zeros(420, 59*n*n, CV_32FC1);
		Mat test_features = Mat::zeros(180, 59*n*n, CV_32FC1);
		Mat train_labels(420, 1, CV_32SC1, cv::Scalar(0));
		Mat test_labels(180, 1, CV_32SC1, cv::Scalar(0));
		
		makeTrainTest_db2(n, train_features, test_features, train_labels, test_labels, path_build);//separa la db2

		//create RandomForest classifier	
		auto rf = cv::ml::RTrees::create();		
		//train RandomForest
		rf->train(train_features,ROW_SAMPLE, train_labels);

		//Predict with RandomForest
		cv::Mat predicted_labels(180, 1, CV_32SC1, cv::Scalar(1));
		
		int sum_vp_y_vn=0;
		for(int i=0; i<test_features.rows; i++)
		{
			int prediction = rf->predict(test_features.row(i));
			predicted_labels.at<int>(i,0)=prediction;
			//Evaluation of predition
			if(test_labels.at<int>(i,0)==predicted_labels.at<int>(i,0))
			{
				sum_vp_y_vn = sum_vp_y_vn+1;
			}
		}
		//Show accuracy
		cout<<"SUMA DE LA DIAGONAL = " << sum_vp_y_vn<<endl;
		cout<<"accuracy RandomForest 3 clases (n=2)  : " << (sum_vp_y_vn/180.0)*100<<endl;
	}
}

int main(void)
{
	cout<<"Porfavor ingrese el path de la carpeta build"<<endl;
	cout<<"                                                                                 "<<endl;
	cout<<"Por ejemplo a mi me funciona con:    /home/luis/Escritorio/tarea4-2019/build/    "<<endl;
	cout<<"                                                                                 "<<endl;
	string path_build;

	cin >> path_build;
	
	cout << "Porfavor presione:" << endl;
	cout << "                                    " <<endl;
	cout << "0 -> Correr TODOS los clasificadores"<<endl;
	cout << "                                    " <<endl;
    cout << "1 -> Para SVM 2 clases con n = 4" << endl;
    cout << "2 -> Para Random Forest 2 clases n = 4" << endl;
	cout << "3 -> Para SVM 3 clases n = 4" << endl;
    cout << "4 -> Para Random Forest 3 clases n = 4" << endl;
    cout << "5 -> Para SVM 2 clases con n = 2" << endl;
    cout << "6 -> Para Random Forest 2 clases n = 2" << endl;
	cout << "7 -> Para SVM 3 clases n = 2" << endl;
    cout << "8 -> Para Random Forest 3 clases n = 2" << endl;
	
	int tipoproblema;
	cin >> tipoproblema;
	
	if(tipoproblema==0)
	{
		for(int t=1; t<9;t++)
		{
			caller(t, path_build); 	
		}
	} else {
		caller(tipoproblema, path_build);
	}
	return 0; 
}