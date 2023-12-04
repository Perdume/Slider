package org.example.Learning;

import java.util.ArrayList;
import java.util.Collections;
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
            AIMoves am = new AIMoves();
            am.getAIMoves(createboard(boardlist.get(i)), newloc(playerloclist.get(i)), newloc(ailoc), new ArrayList<>(), null);
            backtrackingAI bt = new backtrackingAI(am.resultboard, playerloclist.get(i), am.ailoclist, am.result,Depths+1, MaxDepths);
            bt.cal();
            scorelist.add(bt.getoptimizedscore());
        }
        int index = scorelist.indexOf(Collections.min(scorelist));
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
