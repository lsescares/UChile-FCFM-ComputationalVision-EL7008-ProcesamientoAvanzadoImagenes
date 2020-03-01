#define _DEBUG
#include <opencv2/opencv.hpp>
#include "sift.hpp"
#include <iostream>
#include <ctime>

using namespace std;
using namespace cv;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void symmetrymatcher(Mat queryDescriptors, Mat trainDescriptors, std::vector<DMatch>& matches)
{
	matches.clear();
	std::vector<DMatch> matches1, matches2;

	BFMatcher matcher(NORM_L2);

	matcher.match(queryDescriptors, trainDescriptors, matches1);
	matcher.match(trainDescriptors, queryDescriptors, matches2);

	for (vector<DMatch>::const_iterator matchIterator1 = matches1.begin(); matchIterator1 != matches1.end(); ++matchIterator1)
	{
		for (vector<DMatch>::const_iterator matchIterator2 = matches2.begin(); matchIterator2 != matches2.end(); ++matchIterator2)
		{
			if ((*matchIterator1).queryIdx == (*matchIterator2).trainIdx && (*matchIterator2).queryIdx == (*matchIterator1).trainIdx)
			{
				matches.push_back(DMatch((*matchIterator1).queryIdx, (*matchIterator1).trainIdx, (*matchIterator1).distance));
				break;
			}
		}
	}
}

void genTransform(DMatch match, vector<KeyPoint> &keypoints1, vector<KeyPoint> &keypoints2, double &e, double &theta, double &tx, double &ty)
{
	e = keypoints2[match.queryIdx].size/keypoints1[match.trainIdx].size;
	theta = keypoints2[match.queryIdx].angle - keypoints1[match.trainIdx].angle;
	if(theta < double(0.0))
	{
			theta =360+theta;
	}
	theta = theta*M_PI/double(180.0);//pasar grado a radian.
	
	float x_ref, y_ref, x_pru, y_pru;
	x_ref = keypoints1[match.trainIdx].pt.x;
	y_ref = keypoints1[match.trainIdx].pt.y;
	x_pru = keypoints2[match.queryIdx].pt.x;
	y_pru = keypoints2[match.queryIdx].pt.y;

	tx = x_pru - e*(x_ref*cos(theta) - y_ref*sin(theta));
	ty = y_pru - e*(x_ref*sin(theta) + y_ref*cos(theta));
}

int computeConsensus(vector<DMatch> &matches, vector<KeyPoint> &keypoints1, vector<KeyPoint> &keypoints2, vector<int> &selected, double e, double theta, double tx, double ty, double umbralpos)
{
    int cons = 0;
	selected.clear();
	for (int i=0; i<matches.size(); i++)
	{
		float x_ref, y_ref, x_pru, y_pru;
		x_ref = keypoints1[matches[i].trainIdx].pt.x;
		y_ref = keypoints1[matches[i].trainIdx].pt.y;
		x_pru = keypoints2[matches[i].queryIdx].pt.x;
		y_pru = keypoints2[matches[i].queryIdx].pt.y;

		double error, error_x, error_y;
		error_x = e*(cos(theta)*x_ref - sin(theta)*y_ref) + tx - x_pru;
		error_y = e*(sin(theta)*x_ref + cos(theta)*y_ref) + ty - y_pru;
		error = sqrt(pow(error_x,2)+pow(error_y,2));
		
		if(error < umbralpos)
		{
			cons += 1;
			selected.push_back(i);
		}
	}	
	return cons;
}

bool ransac(vector<DMatch> &matches, vector<KeyPoint> &keypoints1, vector<KeyPoint> &keypoints2, vector<DMatch> &accepted)
{
	bool success = false;
	double e=1, theta=0, tx=0, ty=0;
	vector<int> selected;
	vector<int> bestSelected;
	
	// Parametros de Ransac a probar
	double umbralpos = 60;
	int umbralcons = 30;
	int ntrials = 10000;

	vector<int> indices_probados;
	int mejor_concenso = 0;
	int indice_matches_mejor_concenso;

	for(int i=0; i< ntrials; i++)
	{
		int idx_hip = rand()%matches.size();//genera un numero aleatorio entre 0 y matches.size()
		genTransform(matches[idx_hip], keypoints1, keypoints2, e, theta, tx, ty);
		int cantidad_concenso; 
		cantidad_concenso = computeConsensus(matches, keypoints1, keypoints2, selected, e, theta, tx, ty, umbralpos);
		bestSelected.push_back(cantidad_concenso);//guarda los concensos calculados
		indices_probados.push_back(idx_hip);
	}

	for (int j=0; j<indices_probados.size(); j++)
	{
		if(bestSelected[j] > mejor_concenso)
		{
			mejor_concenso = bestSelected[j];
			indice_matches_mejor_concenso= indices_probados[j]; 
		}
	}
	
	if(mejor_concenso > umbralcons)
	{
		genTransform(matches[indice_matches_mejor_concenso], keypoints1, keypoints2, e, theta, tx, ty);
		int cantidad_concenso = computeConsensus(matches, keypoints1, keypoints2, selected, e, theta, tx, ty, umbralpos);//actualizar selected 
		for(int k=0; k<selected.size(); k++)
		{
			accepted.push_back(matches[selected[k]]);
		}
	}
	return false;
}

bool hough4d(vector<DMatch> &matches, vector<KeyPoint> &keypoints1, vector<KeyPoint> &keypoints2, vector<DMatch> &accepted)
{
	int hsize[]={1000, 1000, 1000, 1000};
	SparseMat sm(4, hsize, CV_32F);
	
	// Parametros de Hough
	double dxBin = 60;
	double dangBin = 30 * M_PI / 180;
	int umbralvotos = 5;
	double e=1, theta=0, tx=0, ty=0;

	for(int m =0; m <matches.size(); m++)
	{
		genTransform(matches[m], keypoints1, keypoints2, e, theta, tx, ty);
		int i,j,k,z;
		i=floor((tx/dxBin)+0.5)+500;
		j=floor((ty/dxBin)+0.5)+500;
		k=floor((theta/dangBin)+0.5)+500;
		z=floor((log(e)/log(2.0))+0.5)+500;
		
		int idx[4];
		idx[0] = i;
		idx[1] = j;
		idx[2] = k;
		idx[3] = z;

		sm.ref<float>(idx)++;
	}
	double minvotos, maxvotos;
	int sm_maxvotos_Idx[4], sm_minvotos_Idx[4];
	cv::minMaxLoc(sm, &minvotos, &maxvotos, sm_minvotos_Idx, sm_maxvotos_Idx);//retornar indice de casilla con mas votos
	

	if(maxvotos > umbralvotos)
	{
		int i_accepted = sm_maxvotos_Idx[0];
		int j_accepted = sm_maxvotos_Idx[1];
		int k_accepted = sm_maxvotos_Idx[2];
		int z_accepted = sm_maxvotos_Idx[3];
		
		for(int j=0; j<matches.size();j++)
		{
			genTransform(matches[j], keypoints1, keypoints2, e, theta, tx, ty);
			int i_candidate = floor((tx/dxBin)+0.5)+500;
			int j_candidate = floor((ty/dxBin)+0.5)+500;
			int k_candidate = floor((theta/dangBin)+0.5)+500;
			int z_candidate = floor((log(e)/log(2.0))+0.5)+500;
			if(i_candidate == i_accepted && j_candidate == j_accepted && k_candidate == k_accepted && z_candidate == z_accepted)
			{
				accepted.push_back(matches[j]);
			}

		}
	}
	return true;
}


Mat calcAfin(vector<DMatch>& matches, vector<KeyPoint>& keypoints1, vector<KeyPoint>& keypoints2){
	
	Mat transf = Mat::zeros(2, 3, CV_32FC1);

	if(matches.size()==0)//caso de no encontrar matches aceptados 
	{
		cout<<"Ni RANSAC ni Hough encuentran un conjunto  matches realmente correctos para definir una t. afin entre las 2 imagenes"<<endl;
		return transf;
	}
	Mat A = Mat::zeros(2*matches.size(), 6, CV_32FC1);
	Mat x = Mat::zeros(6, 1, CV_32FC1);
	Mat B = Mat::zeros(2*matches.size(), 1, CV_32FC1);
	
	//'agmentation' A y B
	float x_ref, y_ref, x_pru, y_pru;

	for (int i = 0; i < matches.size(); i++) 
	{
		x_ref = keypoints1[matches[i].trainIdx].pt.x;
		y_ref = keypoints1[matches[i].trainIdx].pt.y;
		x_pru = keypoints2[matches[i].queryIdx].pt.x;
		y_pru = keypoints2[matches[i].queryIdx].pt.y;

		A.at<float>(2*i,0) = x_ref;		
		A.at<float>(2*i,1) = y_ref;			
		A.at<float>(2*i+1,2) = x_ref;
		A.at<float>(2*i+1,3) = y_ref;
		A.at<float>(2*i,4) = 1.0;
		A.at<float>(2*i+1,5) = 1.0;

		B.at<float>(2*i,0) = x_pru;		
		B.at<float>(2*i+1,0) = y_pru;
	}
	x = ((A.t() * A).inv()) * A.t() * B;
	
	transf.at<float>(0,0) = x.at<float>(0,0);
	transf.at<float>(0,1) = x.at<float>(1,0);
	transf.at<float>(1,0) = x.at<float>(2,0);
	transf.at<float>(1,1) = x.at<float>(3,0);
	transf.at<float>(0,2) = x.at<float>(4,0);	
	transf.at<float>(1,2) = x.at<float>(5,0);
	return transf;
}

Mat drawProjAfin(Mat& transf, Mat& input1, Mat& input2){
	Mat output = input1.clone();

	Point si_pru(0.0, input2.rows);		    //s->superior, i->inferior,
	Point sd_pru(input2.cols, input2.rows); //i->izquierda, d->derecha
	Point id_pru(input2.cols, 0.0);         //ii_pru->punto inferior izquierdo de imagen prueba
	Point ii_pru(0.0, 0.0);				
	
	float si_x_ref, si_y_ref, sd_x_ref, sd_y_ref, id_x_ref, id_y_ref, ii_x_ref, ii_y_ref;
	si_x_ref = (transf.at<float>(0,0) * si_pru.x) + (transf.at<float>(0,1) * si_pru.y) + transf.at<float>(0,2);
	si_y_ref = (transf.at<float>(1,0) * si_pru.x) + (transf.at<float>(1,1) * si_pru.y) + transf.at<float>(1,2);
	
	sd_x_ref = (transf.at<float>(0,0) * sd_pru.x) + (transf.at<float>(0,1) * sd_pru.y) + transf.at<float>(0,2);
	sd_y_ref = (transf.at<float>(1,0) * sd_pru.x) + (transf.at<float>(1,1) * sd_pru.y) + transf.at<float>(1,2);
	
	id_x_ref = (transf.at<float>(0,0) * id_pru.x) + (transf.at<float>(0,1) * id_pru.y) + transf.at<float>(0,2);
	id_y_ref = (transf.at<float>(1,0) * id_pru.x) + (transf.at<float>(1,1) * id_pru.y) + transf.at<float>(1,2);
	
	ii_x_ref = (transf.at<float>(0,0) * ii_pru.x) + (transf.at<float>(0,1) * ii_pru.y) + transf.at<float>(0,2);
	ii_y_ref = (transf.at<float>(1,0) * ii_pru.x) + (transf.at<float>(1,1) * ii_pru.y) + transf.at<float>(1,2);
	
	Point si_ref(si_x_ref, si_y_ref);					
	Point sd_ref(sd_x_ref, sd_y_ref);			
	Point id_ref(id_x_ref, id_y_ref);			
	Point ii_ref(ii_x_ref, ii_y_ref);	
	
	line(output, si_ref, sd_ref, Scalar(0, 255,0),1,8,0);
	line(output, sd_ref, id_ref, Scalar(0, 255,0),1,8,0);
	line(output, id_ref, ii_ref, Scalar(0, 255,0),1,8,0);
	line(output, ii_ref, si_ref, Scalar(0, 255,0),1,8,0);
	
	return output;
}

int main(void)
{
	string pru_img_names[] = {"uch006a" , "uch084a","uch098a","uch101a"}; 
	string ref_img_names[] = {"uch006b" , "uch084b","uch098b","uch101b"};
	for (int s=0; s<4; s++)
	{
		Mat input1, input2;
		input1 = imread("im/"+pru_img_names[s]+".jpg"); //img QUERY //img PRU
		input2 = imread("im/"+ref_img_names[s]+".jpg"); //img TRAIN // img REF

		srand(time(NULL)); // Inicializar generador de numeros al azar

		if(input1.empty() || input2.empty()) // No encontro la imagen
		{
			cout<<"Imagen no encontrada"<<endl;
			return 1; // Sale del programa anormalmente
		}

		vector<KeyPoint> keypoints1;
		vector<KeyPoint> keypoints2;
		Mat descriptors1, descriptors2;

		Ptr<DescriptorExtractor> descriptorExtractor = xfeatures2d::SIFT::create(500);
		descriptorExtractor->detectAndCompute(input1, Mat(), keypoints1, descriptors1);
		descriptorExtractor->detectAndCompute(input2, Mat(), keypoints2, descriptors2);

		vector<DMatch> matches;
		symmetrymatcher(descriptors2, descriptors1, matches);

		vector<DMatch> acceptedRansac;
		ransac(matches, keypoints1, keypoints2, acceptedRansac);

		vector<DMatch> acceptedHough;
		hough4d(matches, keypoints1, keypoints2, acceptedHough);

		// drawing the results
		Mat output1, output2;
		drawKeypoints(input1, keypoints1, output1);
		drawKeypoints(input2, keypoints2, output2);
		
		Mat keypoints2imagenes;
		vector<DMatch> matches_vacios;
		drawMatches(input2, keypoints2, input1, keypoints1, matches_vacios, keypoints2imagenes);
		imwrite(pru_img_names[s]+"_keypoints.png", keypoints2imagenes);
	
		Mat img_matches;
		drawMatches(input2, keypoints2, input1, keypoints1, matches, img_matches);
		imwrite(pru_img_names[s]+"_matches_symmetrymatcher.png", img_matches);//graficar matches resultados de symmetrymatcher
		
		Mat img_accepted_RANSAC;
		drawMatches(input2, keypoints2, input1, keypoints1, acceptedRansac, img_accepted_RANSAC);
		imwrite(pru_img_names[s]+"_acceptedRansac.png", img_accepted_RANSAC);
		
		Mat img_accepted_Hough;
		drawMatches(input2, keypoints2, input1, keypoints1, acceptedHough, img_accepted_Hough);
		imwrite(pru_img_names[s]+"_acceptedHough.png", img_accepted_Hough);
		
		Mat transf_RANSAC = calcAfin(acceptedRansac, keypoints1, keypoints2);
		Mat img_proj_RANSAC = drawProjAfin(transf_RANSAC, input2, input1);
		imwrite(pru_img_names[s]+"_proj_RANSAC.png", img_proj_RANSAC);

		Mat transf_HOUGH = calcAfin(acceptedHough, keypoints1, keypoints2);
		Mat img_proj_HOUGH = drawProjAfin(transf_HOUGH, input2, input1);
		imwrite(pru_img_names[s]+"_proj_HOUGH.png", img_proj_HOUGH);
	}
	cout<<"*************************************************************************************"<<endl;
	cout<<"   El codigo funciono y guardo todas las imagenes resultantes en la carpeta bluid"<<endl;	
	cout<<"*************************************************************************************"<<endl;
	//waitKey(0);
	return 0; // Sale del programa normalmente
}