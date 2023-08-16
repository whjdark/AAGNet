# -*- coding: utf-8 -*-
import argparse


from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.StlAPI import StlAPI_Writer



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--step_path", type=str, required=True, help="Path to load the step files from")
    parser.add_argument("--output", type=str, required=True, help="Path to the save intermediate brep data")
    args = parser.parse_args()

    # Create a reader object
    reader = STEPControl_Reader()

    # Read the STEP file
    status = reader.ReadFile(args.step_path)

    # Check if the file was read successfully
    if status == 1:
        # Transfer the shape object
        reader.TransferRoot()
        shape = reader.Shape()
        mesh = BRepMesh_IncrementalMesh(shape, 0.1)

        # Create a writer object
        writer = StlAPI_Writer()
        
        # Write the shape object to an STL file
        err = writer.Write(mesh.Shape(), args.output)

        # Print a success message
        print(err)
    else:
        # Print an error message
        print('Could not read the STEP file.')

    