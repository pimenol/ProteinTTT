#!/bin/bash

PROTEIN_ID=$1

cp "./data/bfvd/predicted_structures/${PROTEIN_ID}_ttt.pdb" "./copy/${PROTEIN_ID}_ttt.pdb"
cp "./data/bfvd/esm_fold/${PROTEIN_ID}.pdb" "./copy/${PROTEIN_ID}_esm.pdb"
cp "/scratch/project/open-35-8/antonb/bfvd/bfvd/${PROTEIN_ID}.pdb" "./copy/${PROTEIN_ID}_af2.pdb"

echo "Files copied for $PROTEIN_ID."