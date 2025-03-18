#include "Engine/Search.h"
#include <iostream>
#include "Engine/MoveGen.h"
#include "Engine/Move.h"
#include "Engine/Evaluation.h"
#include "Engine/Magic.h"
#include <algorithm>
#include "Engine/HashTable.h"
#include "nnue_loader.h"


const int Search::INFINITE = 30000;
static const int MATE_SCORE = 29999;

const int Search::ASPIRATION_START_DEPTH = 5;
const int Search::REDUCTION_LIMIT = 3;
const int Search::REDUCTION_DEPTH_LATE = 6;
const int Search::IID_REDUCTION = 2;

int Search::VICTIM_SCORES[14] = {
		0, 0, 
		Evaluation::PAWN_VAL, Evaluation::PAWN_VAL, Evaluation::KNIGHT_VAL, Evaluation::KNIGHT_VAL,
		Evaluation::BISHOP_VAL, Evaluation::BISHOP_VAL, Evaluation::ROOK_VAL, Evaluation::ROOK_VAL,
		Evaluation::QUEEN_VAL, Evaluation::QUEEN_VAL, Evaluation::KING_VAL, Evaluation::KING_VAL
		};

int Search::MVV_VLA_SCORES[14][14];
		
void Search::initHeuristics(){
	for (int v = 0; v < 14; v++){
		for (int a = 0; a < 14; a++){
			MVV_VLA_SCORES[v][a] = VICTIM_SCORES[v] + 6 - (VICTIM_SCORES[a]/100);
		}
	}
}

Search::Search() : model(NNUELoader::getInstance()) {
	model.loadWeights("D:\\cpp_projs\\NNUE\\NNUE\\weights\\weights_2_128.npz");
}

void Search::refreshNNUE(){
	model.setAccumulator(board);
}

Search::Search(Board b, SearchInfo i) : board(b), info(i), model(NNUELoader::getInstance()) {
	model.setAccumulator(board);
}

void Search::stop(){
	info.stopped = true;
}

U64 Search::getTime(){
	U64 t = (U64)(std::chrono::system_clock::now().time_since_epoch()/std::chrono::milliseconds(1));
	return t;
}

void Search::checkUp(SearchInfo& info){
	if (info.timeSet && (Search::getTime() > info.stopTime)){
		info.stopped = true;
	}
}

std::vector<MoveScore> Search::moveScore(Move::MAX_LEGAL_MOVES);
void Search::orderMoves(Board& board, MoveList& moves, int pvMove) {

	int side = board.state.currentPlayer;
	for (int i = 0; i < moves.size(); i++) {
		int mv = moves.get(i);
		int capt = Move::captured(mv);
		int promo = Move::promoteTo(mv);
		int from = Move::from(mv);
		int attacker = board.board[from];

		//PV override (highest priority)
		if (mv == pvMove) {
			moveScore[i] = MoveScore(mv, PV_BONUS);
			continue;
		}

		if (capt > 0) {
			int to = Move::to(mv);
			int attacker = board.board[from];
			moveScore[i] = MoveScore(mv, MVV_VLA_SCORES[capt][attacker] + CAPT_BONUS);
		}
		else if (Move::isEP(mv)) {
			moveScore[i] = MoveScore(mv, MVV_VLA_SCORES[Board::WHITE_PAWN][Board::BLACK_PAWN] + CAPT_BONUS);
		}
		else {
			if (board.searchKillers[0][board.ply] == mv) {
				moveScore[i] = MoveScore(mv, KILLER_BONUS_0);
			}
			else if (board.searchKillers[1][board.ply] == mv) {
				moveScore[i] = MoveScore(mv, KILLER_BONUS_1);
			}
			else {
				int piece = board.board[Move::from(mv)];
				int to = Move::to(mv);
				moveScore[i] = MoveScore(mv, board.searchHistory[piece][to]);
			}
		}

		if (promo) {
			moveScore[i].score += PROMO_BONUS + abs(Evaluation::PIECE_VALUES[promo]);
		}
	}

	std::sort(moveScore.begin(), moveScore.begin() + moves.size(), std::less<MoveScore>());

	for (int i = 0; i < moves.size(); i++) {
		moves.set(i, moveScore[i].move);
	}
}

void Search::clearSearch(){
	for (int i = 0; i < 14; i++){
		for (int j = 0; j < 64; j++){
			board.searchHistory[i][j] = 0;
		}
	}

	//clear killers
	for (int i = 0; i < 2; i++){
		for (int j = 0; j < Board::MAX_DEPTH; j++){
			board.searchKillers[i][j] = 0;
		}
	}

	board.hashTable->overWrite = 0;
	board.hashTable->hit = 0;	
	board.hashTable->cut = 0;	
	board.ply = 0;
	
	info.stopped = false;
	info.nodes = 0;
	info.fh = 0.0f;
	info.fhf = 0.0f;
	info.nullCut = 0;
}


int Search::search(bool verbose){
	int bestMove = Move::NO_MOVE;
	int bestScore = 0;
	int pvMoves = 0;
	
	clearSearch();
	refreshNNUE();

	//iterative deepening
	for (int currentDepth = 1; currentDepth <= info.depth; currentDepth++){
		//bestScore = alphaBeta(-INFINITE, INFINITE, currentDepth, true);
		bestScore = aspirationWindow(&board, currentDepth, bestScore);

		//check stop
		if (info.stopped){
			break;
		}

		pvMoves = HashTable::getPVLine(currentDepth, board);
		bestMove = board.pvArray[0];
		
		if (verbose){
			printf("info score cp %d depth %d nodes %llu time %llu pv",
					bestScore, currentDepth, info.nodes, Search::getTime() - info.startTime);			
			
			for (int i = 0; i < pvMoves; i++){
				int m = board.pvArray[i];
				std::cout << " " <<  Move::toLongNotation(m);
			}
			printf("\n");
		}
		
		/*printf("Hits: %d  Overwrite: %d  NewWrite: %d  Cut: %d\nOrdering: %.2f  NullCut:%d\n",
		 	board.hashTable->hit, board.hashTable->overWrite, board.hashTable->newWrite, board.hashTable->cut,
		 	(info.fhf/info.fh)*100, info.nullCut);	*/	
	}

	if (verbose)
		std::cout << "bestmove " << Move::toLongNotation(bestMove) << std::endl;
	return bestMove;
}

int Search::aspirationWindow(Board* board, int depth, int score) {
	if (depth <= ASPIRATION_START_DEPTH) {
		return alphaBeta(-MATE_SCORE, MATE_SCORE, depth, true);
	}

	int delta = 15;
	int alpha = std::max(-MATE_SCORE, score - delta);
	int beta = std::min(MATE_SCORE, score + delta);
	int f = score;

	while (true) {
		f = alphaBeta(alpha, beta, depth, true);

		if (info.stopped) {
			return 0; // Or return f;  Either is fine.
		}

		if (f > alpha && f < beta) {
			return f; // Success within the window
		}

		if (f <= alpha) {
			alpha = std::max(-MATE_SCORE, alpha - delta);
		}

		if (f >= beta) {
			beta = std::min(MATE_SCORE, beta + delta);
		}

		delta += delta / 2;

		//Prevent infinite loops when close to mate.
		if (delta > 800) {
			return alphaBeta(-MATE_SCORE, MATE_SCORE, depth, true);
		}
	}
}

static const int FUTIL_MARGIN[4] = { 0, 225, 350, 550 };
static const int RAZOR_MARGIN[4] = { 0, 325, 345, 395 };

int Search::alphaBeta(int alpha, int beta, int depth, bool doNull) {

	if ((info.nodes & 2047) == 0) {
		checkUp(info);
	}

	int side = board.state.currentPlayer;
	int opp = side ^ 1;

	//Check extension (put this BEFORE draw/repetition check)
	bool atCheck = MoveGen::isSquareAttacked(&board, board.kingSQ[side], opp);
	if (atCheck)
		depth++;

	if ((board.state.halfMoves >= 100 || board.isRepetition()) && board.ply) {
		return 0;
	}

	if (board.ply > Board::MAX_DEPTH - 1) {
		//return Evaluation::evaluate(board, side);
		return model.computeOutput(side == Board::WHITE ? WHITE_NNUE : BLACK_NNUE);
	}


	//Quiesce
	if (depth <= 0) {
		return Quiescence(alpha, beta);
	}

	info.nodes++;

	// --- Hash Table Probe ---
	int score = -INFINITE;
	int pvMove = Move::NO_MOVE;
	int hashFlag = HFALPHA; // Assume we'll set an alpha bound

	if (HashTable::probeHashEntry(board, &pvMove, &score, alpha, beta, depth)) {
		board.hashTable->cut++;
		// Adjust score for mate distance
		if (score > ISMATE) score -= board.ply;
		else if (score < -ISMATE) score += board.ply;

		return score;
	}
	// --- End Hash Table Probe ---
	// Internal Iterative Deepening
	if (!pvMove && depth >= 5) {
		int iidDepth = depth - IID_REDUCTION;
		alphaBeta(alpha, beta, iidDepth, true);
		if (info.stopped) return 0;
		HashTable::probeHashEntry(board, &pvMove, &score, alpha, beta, iidDepth);
	}

	//Static Evaluation (for pruning) - only if NOT in check
	int static_eval = 0;
	bool static_set = false;

	if (!atCheck) {
		//static_eval = Evaluation::evaluate(board, side);		
		static_eval = model.computeOutput(side == Board::WHITE ? WHITE_NNUE : BLACK_NNUE);
		static_set = true;
	}


	//Eval pruning (Static Null Move Pruning) - Before Null Move!
	if (!atCheck && !pvMove && depth < 3 && abs(beta - 1) > -INFINITE + 100) {
		int eval_margin = 120 * depth;
		if (static_eval - eval_margin >= beta)
			return static_eval - eval_margin;
	}

	// --- Null Move Pruning ---
	if (doNull && !atCheck && board.ply > 0 && depth >= 3) {
		// Ensure we have enough material for a null move
		if (board.material[side] > ENDGAME_MAT)
		{
			BoardState undo_null = board.makeNullMove();
			int R = 2 + depth / 6; // Adjust reduction based on depth
			score = -alphaBeta(-beta, -beta + 1, depth - R - 1, false);
			board.undoNullMove(undo_null);

			if (info.stopped) return 0;

			// Adjust score for mate distance
			if (score > ISMATE) score -= board.ply;
			else if (score < -ISMATE) score += board.ply;

			if (score >= beta) {		
				info.nullCut++;
				return score;
			}
		}
	}
	// --- End Null Move Pruning ---


	//Futility pruning
	bool f_prune = false;
	if (depth <= 3 && !atCheck && abs(alpha) < 9000 && static_set) {
		if (static_eval + FUTIL_MARGIN[depth] <= alpha)
			f_prune = true;
	}

	// --- Move Generation and Ordering ---
	MoveList moves;
	MoveGen::pseudoLegalMoves(&board, side, moves, atCheck);
	orderMoves(board, moves, pvMove);

	int legal = 0;
	int oldAlpha = alpha;
	int bestMove = Move::NO_MOVE;
	score = -INFINITE;
	int bestScore = -INFINITE;

	// --- Main Search Loop ---
	for (int i = 0; i < moves.size(); i++) {
		int currentMove = moves.get(i);

		FeatureChanges changes = FeatureExtractor::moveDiffFeatures(board, currentMove);
		model.updateAccumulator(changes);
		BoardState undo = board.makeMove(currentMove);

		//NNUE needs undo???
		if (!undo.valid) {
			model.updateAccumulatorUndo(changes);
			continue;
		}

		legal++;
		bool oppAtCheck = MoveGen::isSquareAttacked(&board, board.kingSQ[opp], side);

		//Futility pruning
		if (f_prune && legal > 0 && !Move::captured(currentMove) && !Move::promoteTo(currentMove) && !oppAtCheck) {
			model.updateAccumulatorUndo(changes);
			board.undoMove(currentMove, undo);
			continue;
		}

		bool doReduce = false;
		// --- Late Move Reduction (LMR) ---
		if (depth >= REDUCTION_LIMIT && legal > REDUCTION_LIMIT && !atCheck && !oppAtCheck &&
			Move::captured(currentMove) == 0 && Move::promoteTo(currentMove) == 0 &&
			currentMove != board.searchKillers[0][board.ply] &&
			currentMove != board.searchKillers[1][board.ply])
		{
			int reduce = (legal > REDUCTION_DEPTH_LATE) ? 2 : 1;
			doReduce = true;
			score = -alphaBeta(-beta, -alpha, depth - 1 - reduce, true);
		}
		else {
			// Full-depth search (or PVS first move)
			score = -alphaBeta(-beta, -alpha, depth - 1, true);
		}

		// --- PVS Re-search ---
		if (doReduce && score > alpha) {
			score = -alphaBeta(-beta, -alpha, depth - 1, true); // Re-search at full depth
		}
		// --- End LMR and PVS ---

		model.updateAccumulatorUndo(changes);
		board.undoMove(currentMove, undo);

		if (info.stopped) {
			return 0;
		}

		// --- Update Best Score and Alpha ---
		if (score > bestScore) {
			bestScore = score;
			bestMove = currentMove;

			if (score > alpha) {
				alpha = score;
				hashFlag = HFEXACT; // We've found an exact score

				if (score >= beta) {
					// --- Beta Cutoff ---
					if (legal == 1)
						info.fhf++;
					info.fh++;

					// --- Killer Heuristic --- (Only on quiet moves)
					if (Move::captured(currentMove) == 0 && !Move::isEP(currentMove)) {
						board.searchKillers[1][board.ply] = board.searchKillers[0][board.ply];
						board.searchKillers[0][board.ply] = currentMove;
					}
					// --- End Killer Heuristic ---

					// --- History Heuristic --- (Only on quiet moves)
					if (Move::captured(currentMove) == 0 && !Move::isEP(currentMove) && !Move::isCastle(currentMove)) {
						int piece = board.board[Move::from(currentMove)];
						board.searchHistory[piece][Move::to(currentMove)] += depth * depth;
					}
					// --- End History Heuristic ---
					//Adjust score for mate distance
					if (bestScore > ISMATE) bestScore -= board.ply;
					else if (bestScore < -ISMATE) bestScore += board.ply;
					HashTable::storeHashEntry(board, bestMove, bestScore, HFBETA, depth);
					return beta;
					// --- End Beta Cutoff ---
				}
			}
		}
	}

	// --- Mate/Stalemate Detection ---
	if (legal == 0) {
		if (atCheck) {
			//Adjust score for mate distance
			return -MATE_SCORE + board.ply;
		}
		else {
			return 0;
		}
	}
	// --- End Mate/Stalemate Detection ---

	// --- Hash Table Store ---
	 //Adjust score for mate distance
	if (bestScore > ISMATE) bestScore -= board.ply;
	else if (bestScore < -ISMATE) bestScore += board.ply;

	HashTable::storeHashEntry(board, bestMove, bestScore, hashFlag, depth);
	// --- End Hash Table Store ---

	return alpha;
}

int Search::Quiescence(int alpha, int beta) {
	assert(alpha < beta);

	//check time
	if ((info.nodes & 2047) == 0) {
		checkUp(info);
	}

	info.nodes++;

	if (board.state.halfMoves >= 100 || board.isRepetition()) {
		return 0;
	}

	int side = board.state.currentPlayer;
	int opp = side ^ 1;
	bool atCheck = MoveGen::isSquareAttacked(&board, board.kingSQ[side], opp);

	if (board.ply > Board::MAX_DEPTH - 1) {
		//return Evaluation::evaluate(board, side);		
		return model.computeOutput(side == Board::WHITE ? WHITE_NNUE : BLACK_NNUE);
	}

	// --- Stand Pat ---
	//int stand_pat = Evaluation::evaluate(board, side);	
	int stand_pat = model.computeOutput(side == Board::WHITE ? WHITE_NNUE : BLACK_NNUE);

	if (!atCheck) {
		if (stand_pat >= beta) {
			return beta;
		}
		if (stand_pat > alpha) {
			alpha = stand_pat;
		}
	}
	// --- End Stand Pat ---

	MoveList moves;
	// Generate moves (only captures and promotions if not in check)
	if (atCheck) {
		U64 occup = board.bitboards[Board::WHITE] | board.bitboards[Board::BLACK];
		MoveGen::getEvasions(&board, side, moves, occup);
	}
	else {
		MoveGen::pseudoLegalCaptureMoves(&board, side, moves);
		MoveGen::pawnPromotions(&board, side, moves, true); // Only generate promotions to queen
	}

	orderMoves(board, moves, Move::NO_MOVE); // Order captures

	int legal = 0;
	int oldAlpha = alpha; // Store the original alpha value
	int score = -INFINITE;

	// --- Loop through Captures ---
	for (int i = 0; i < moves.size(); i++) {
		int currentMove = moves.get(i);
		int capt = Move::captured(currentMove);
		int promo = Move::promoteTo(currentMove);

		// --- Delta Pruning --- (Only if not in check and not a promotion)
		if (!atCheck && promo == 0 &&
			(stand_pat + abs(Evaluation::PIECE_VALUES[capt]) + 200 < alpha) &&
			(board.material[opp] - Evaluation::KING_VAL - abs(Evaluation::PIECE_VALUES[capt]) > ENDGAME_MAT)) {
			continue;
		}
		// --- End Delta Pruning ---

		 // --- SEE Pruning --- (Only if not in check and not a promotion)
		if (!atCheck && !promo && isBadCapture(board, currentMove, side)) {
			continue;
		}
		// --- End SEE Pruning ---

		FeatureChanges changes = FeatureExtractor::moveDiffFeatures(board, currentMove);
		model.updateAccumulator(changes);
		BoardState undo = board.makeMove(currentMove);

		if (!undo.valid) {
			model.updateAccumulatorUndo(changes);
			continue;
		}

		legal++;

		score = -Quiescence(-beta, -alpha);

		model.updateAccumulatorUndo(changes);
		board.undoMove(currentMove, undo);

		if (info.stopped) {
			return 0;
		}

		if (score > alpha) {
			if (score >= beta) {
				if (legal == 1)
					info.fhf++;
				info.fh++;
				return beta;
			}
			alpha = score;
		}
	}

	// --- Check for Checkmate/Stalemate ---
	if (legal == 0 && atCheck) {
		return -MATE_SCORE + board.ply; // Adjust for mate distance
	}
	// --- End Check for Checkmate/Stalemate ---

	return alpha;
}

bool Search::isBadCapture(const Board& board, int move, int side){
	int from = Move::from(move);	
	int to = Move::to(move);	
	int attacker = board.board[from];
	int target = board.board[to];

	return Search::see(&board, to, target, from, attacker) < 0;
}

U64 getLeastValuablePiece(const Board* board, U64 attadef, int side, int& piece){
	for (piece = Board::PAWN + side; piece <= Board::KING + side; piece+= 2){
		U64 subset = attadef & board->bitboards[piece];
		if (subset)
			//return subset & -subset;
			return subset & (~subset + 1); 
	}
	return 0;
}


U64 considerXrays(const Board* board, U64 occu, U64 attackdef, int sq) {
	int color = board->state.currentPlayer;
	U64 rookQueens = board->bitboards[Board::WHITE_ROOK] | board->bitboards[Board::WHITE_QUEEN] |
					 board->bitboards[Board::BLACK_ROOK] | board->bitboards[Board::BLACK_QUEEN];

    U64 bishopQueens = board->bitboards[Board::WHITE_BISHOP] | board->bitboards[Board::WHITE_QUEEN] |
					   board->bitboards[Board::BLACK_BISHOP] | board->bitboards[Board::BLACK_QUEEN];


	U64 att = (Magic::rookAttacksFrom(occu, sq) & rookQueens) | (Magic::bishopAttacksFrom(occu, sq) & bishopQueens);
	return att & occu;
}

//54.67% +/- 1.51% only at quiesce 2857 games
int Search::see(const Board* board, int toSq, int target, int fromSq, int aPiece){
	int gain[32];
	int d = 0;
	int color = board->state.currentPlayer;
	U64 mayXray = board->bitboards[Board::WHITE_PAWN] | board->bitboards[Board::BLACK_PAWN] | 
				  board->bitboards[Board::WHITE_BISHOP] | board->bitboards[Board::BLACK_BISHOP] |
				  board->bitboards[Board::WHITE_ROOK] | board->bitboards[Board::BLACK_ROOK] |
				  board->bitboards[Board::WHITE_QUEEN] | board->bitboards[Board::BLACK_QUEEN];		

	U64 fromSet = (BitBoardGen::ONE << fromSq);
	U64 occup = board->bitboards[Board::WHITE] | board->bitboards[Board::BLACK];
	U64 attadef = MoveGen::attackers_to(board, toSq, Board::WHITE) | MoveGen::attackers_to(board, toSq, Board::BLACK);	
	gain[d] = abs(Evaluation::PIECE_VALUES[target]);

	do {
		d++;		
		gain[d] = abs(Evaluation::PIECE_VALUES[aPiece]) - gain[d - 1];

		if(std::max(-gain[d - 1], gain[d]) < 0) {
            break;
        }
        attadef^= fromSet;
        occup^= fromSet;

        if(fromSet & mayXray) {
            attadef|= considerXrays(board, occup, attadef, toSq);
        }                

        color = !color;
        fromSet = getLeastValuablePiece(board, attadef, color, aPiece);        
	} while(fromSet);

	while (--d)  {
        gain[d - 1]= -std::max(-gain[d - 1], gain[d]);
    }
    return gain[0];
}




