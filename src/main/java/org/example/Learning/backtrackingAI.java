package org.example.Learning;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

import static org.example.util.createboard;
import static org.example.util.newloc;

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
            for (int i=0; i<boardlist.size();i++) {
                PlayerMoves am = new PlayerMoves();
                am.getPlayerMoves(createboard(boardlist.get(i)), newloc(playerloc), newloc(ailoclist.get(i)), new ArrayList<>(), null);
                backtrackingPlayer bt = new backtrackingPlayer(am.resultboard, am.playerloclist, ailoclist.get(i), am.result,Depths, MaxDepths);
                bt.cal();
                scorelist.add(bt.getoptimizedscore());
            }
            System.out.println("Depth-5: " + scorelist);
            int index = scorelist.indexOf(Collections.max(scorelist));
            finboard = boardlist.get(index);
            finplayerloc = playerloc;
            finailoc = ailoclist.get(index);
            score = scorelist.get(index);
            finmove = movelist.get(index);
        }
        else{
            Minimax mn = new Minimax(true, boardlist, (new ArrayList<>(Collections.singleton(playerloc))).stream().flatMap(item -> Collections.nCopies(boardlist.size(), item).stream()).collect(Collectors.toList()), ailoclist);
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
