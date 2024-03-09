package org.example.Learning;

import org.example.util;

import java.util.*;

public class LocationSurrounding {

    public List<List<String>> result = new ArrayList<>(); //Solve#2 List
    public List<String> Fin = new ArrayList<>();
    List<int[][]> resultboard = new ArrayList<>();
    List<int[]> enemyloclist = new ArrayList<>();
    int[] playerloc;

    public void Clearing(){
        List<List<String>> Tempresult = new ArrayList<>(result);
        List<String> TempFin = new ArrayList<>(Fin);
        for(int i=0;i<Fin.size();i++){
            int count = TempFin.size();
            if(i < count) {
                for (int j = 0; j < Fin.size(); j++) {
                    if(j< count) {
                        if(Objects.equals(TempFin.get(i), TempFin.get(j))){
                            if(result.get(i).size() < result.get(j).size()){
                                Tempresult.remove(i);
                                TempFin.remove(i);
                                count--;
                            }
                            else{
                                Tempresult.remove(j);
                                TempFin.remove(j);
                                count--;
                            }
                        }
                    }
                }
            }
        }
        System.out.print(result.size() - Tempresult.size() +", ");
        result = Tempresult;
        Fin = TempFin;
    }

    public List<List<String>> markVisited(int[][] list, int[] loc, int[] enemyloc, Set<List<Integer>> invlocs) {
        List<List<String>> movements = new ArrayList<>();
        VirtualBoard vb = new VirtualBoard(util.createboard(list), util.newloc(loc), util.newloc(enemyloc));
        // Check if the position is within the bounds of the array
        int[] dx = {-1, 1, 0, 0};
        int[] dy = {0, 0, -1, 1};
        String[] moveSymbols = {"w", "s", "a", "d"};

        for (int i = 0; i < moveSymbols.length; i++) {
            int[] move = {loc[0] + dx[i], loc[1] + dy[i]};

            if (canMove(move, enemyloc) && !invlocs.contains(Arrays.asList(move[0], move[1]))) {
                if (list[move[0]][move[1]] == 0) {
                    List<String> movement = new ArrayList<>();
                    vb.VirtualMove(dx[i], dy[i]);
                    movement.add(moveSymbols[i]);
                    movements.add(movement);
                    resultboard.add(vb.board);
                    enemyloclist.add(vb.enemyLoc);
                    playerloc = loc;
                    result.add(movement);
                    Fin.add(vb.playerLoc[0]+vb.playerLoc[1]+moveSymbols[i]);
                } else {
                    invlocs.add(Arrays.asList(loc[0], loc[1]));
                    List<List<String>> subMovements = markVisited(list, move, enemyloc, invlocs);
                    for (List<String> subMovement : subMovements) {
                        subMovement.add(0, moveSymbols[i]);
                        movements.add(subMovement);
                    }
                }
            }
        }

        return movements;
    }

    private boolean canMove(int[] loc, int[] enemyloc) {
        int row = loc[0];
        int col = loc[1];
        return row >= 0 && row <= 6 && col >= 0 && col <= 6 && (col != enemyloc[1] || row != enemyloc[0]);
    }
}

