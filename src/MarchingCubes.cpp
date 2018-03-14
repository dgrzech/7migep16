/* sys headers */
#include <cstring>
#include <iostream>
#include <string>

/* vtk */
#include <vtkDICOMImageReader.h>
#include <vtkImageData.h>
#include <vtkMarchingCubes.h>
#include <vtkSmartPointer.h>
#include <vtkSphereSource.h>
#include <vtkVersion.h>
#include <vtkVoxelModeller.h>

#include <vtkActor.h>
#include <vtkPolyDataMapper.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>

#include <vtkPLYWriter.h>

/*
 * extracts iso-surface from dicom data and saves the output to a .ply file
 *
 * need to provide a folder containing the series and specify an isovalue for
 * the surface
 */
int main(int argc, char *argv[]) {
	vtkSmartPointer<vtkImageData> volume =
	    vtkSmartPointer<vtkImageData>::New();
	double isoValue;
	if (argc < 3) {
		return EXIT_FAILURE;
	} else {
		vtkSmartPointer<vtkDICOMImageReader> reader =
		    vtkSmartPointer<vtkDICOMImageReader>::New();
		reader->SetDirectoryName(argv[1]);
		reader->Update();
		volume->DeepCopy(reader->GetOutput());
		isoValue = atof(argv[2]);
	}

	vtkSmartPointer<vtkMarchingCubes> surface =
	    vtkSmartPointer<vtkMarchingCubes>::New();

#if VTK_MAJOR_VERSION <= 5
	surface->SetInput(volume);
#else
	surface->SetInputData(volume);
#endif
	surface->ComputeNormalsOn();
	surface->SetValue(0, isoValue);

	std::string fileName(argv[1]);
	fileName = fileName + "/output.ply";
	vtkSmartPointer<vtkPLYWriter> plyWriter =
	    vtkSmartPointer<vtkPLYWriter>::New();
	plyWriter->SetFileName(fileName.c_str());
	plyWriter->SetInputConnection(surface->GetOutputPort());
	plyWriter->Write();

	vtkSmartPointer<vtkRenderer> renderer =
	    vtkSmartPointer<vtkRenderer>::New();
	renderer->SetBackground(.1, .2, .3);

	vtkSmartPointer<vtkRenderWindow> renderWindow =
	    vtkSmartPointer<vtkRenderWindow>::New();
	renderWindow->AddRenderer(renderer);
	vtkSmartPointer<vtkRenderWindowInteractor> interactor =
	    vtkSmartPointer<vtkRenderWindowInteractor>::New();
	interactor->SetRenderWindow(renderWindow);

	vtkSmartPointer<vtkPolyDataMapper> mapper =
	    vtkSmartPointer<vtkPolyDataMapper>::New();
	mapper->SetInputConnection(surface->GetOutputPort());
	mapper->ScalarVisibilityOff();

	vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
	actor->SetMapper(mapper);

	renderer->AddActor(actor);

	renderWindow->Render();
	interactor->Start();
	return EXIT_SUCCESS;
}
