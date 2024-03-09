package org.example.Learning;

import org.example.util;

import java.util.*;
import java.util.stream.Collectors;

import static org.example.util.*;

public class backtrackingAI {
    List<int[][]> boardlist;
    int[] playerloc;
    List<int[]> ailoclist;
    List<List<String>> movelist;
    int Depths;
    int MaxDepths;
    int[][] finboard;
    int[] finplayerloc;
    int[] finailoc;
    List<String> finmove;
    Double score;


    public backtrackingAI(List<int[][]> boards, int[] plllocs, List<int[]> ailocs,List<List<String>> moves, int depths, int maxDepths){
        boardlist = boards;
        playerloc = plllocs;
        ailoclist = ailocs;
        movelist = moves;
        Depths = depths;
        MaxDepths = maxDepths;
    }

    public void cal(){
        List<Double> scorelist = new ArrayList<>();
        if(Depths != MaxDepths) {
            for (int i = 0; i < boardlist.size(); i++) {
                if(!util.isEquallocation(ailoclist.get(i), new int[]{3, 3})){
                    LocationSurrounding ls = new LocationSurrounding();
                    ls.markVisited(createboard(boardlist.get(i)), newloc(playerloc), newloc(ailoclist.get(i)), new HashSet<>());
                    ls.Clearing();
                    backtrackingPlayer bt = new backtrackingPlayer(ls.resultboard, util.copiedloc(playerloc, ls.resultboard.size()), ailoclist.get(i), ls.result, Depths, MaxDepths);
                    bt.cal();
                    scorelist.add(bt.getoptimizedscore());
                }
                else{
                    scorelist.add(1000.0);
                }
            } // set score
            int index = scorelist.indexOf(Collections.max(scorelist));
            finboard = boardlist.get(index);
            finplayerloc = playerloc;
            finailoc = ailoclist.get(index);
            score = scorelist.get(index);
            finmove = movelist.get(index);
        }
        else{
            Minimax mn = new Minimax(true, boardlist, copiedloc(playerloc, boardlist.size()), ailoclist);
            scorelist = mn.calculate();
            int index = scorelist.indexOf(Collections.max(scorelist));
            finboard = boardlist.get(index);
            finplayerloc = playerloc;
            finailoc = ailoclist.get(index);
            score = scorelist.get(index);
        }
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
    public List<String> getoptimizedmove(){
        return finmove;
    }
    public Double getoptimizedscore(){
        return score;
    }
}
