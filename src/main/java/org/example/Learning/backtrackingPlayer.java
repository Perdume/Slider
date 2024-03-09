package org.example.Learning;

import org.example.util;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;

import static org.example.util.createboard;
import static org.example.util.newloc;
public class backtrackingPlayer {
    List<int[][]> boardlist;
    List<int[]> playerloclist;
    int[] ailoc;
    List<List<String>> movelist;
    int Depths;
    int MaxDepths;
    int[][] finboard;
    int[] finplayerloc;
    int[] finailoc;
    Double score;


    public backtrackingPlayer(List<int[][]> boards, List<int[]> plllocs, int[] ailocs,List<List<String>> moves, int depths, int maxDepths){
        boardlist = boards;
        playerloclist = plllocs;
        ailoc = ailocs;
        movelist = moves;
        Depths = depths;
        MaxDepths = maxDepths;
    }
    public void cal(){
        List<Double> scorelist= new ArrayList<>();
        for (int i=0; i<boardlist.size();i++) {
            if(!util.isEquallocation(playerloclist.get(i), new int[]{3, 3})) {
                LocationSurrounding ls = new LocationSurrounding();
                ls.markVisited(createboard(boardlist.get(i)), newloc(playerloclist.get(i)), newloc(ailoc), new HashSet<>());
                ls.Clearing();
                backtrackingAI bt = new backtrackingAI(ls.resultboard, playerloclist.get(i), ls.enemyloclist, ls.result, Depths + 1, MaxDepths);
                bt.cal();
                scorelist.add(bt.getoptimizedscore());
            }
            else{
                scorelist.add(-1000.0);
            }
        } // setscore
        int index = scorelist.indexOf(Collections.max(scorelist));
        finboard = boardlist.get(index);
        finplayerloc = playerloclist.get(index);
        finailoc = ailoc;
        score = scorelist.get(index);
    }
    public int[][] getoptimizedboard(){
        return finboard;
    }
    public int[] getoptimizedplayerloc(){
        return finplayerloc;
    }
    public int[] getoptimizedailoc(){
        return finailoc;
    }
    public Double getoptimizedscore(){
        return score;
    }

}
