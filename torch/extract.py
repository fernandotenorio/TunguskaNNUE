import csv

def extract_chess_data(file_path):
    data = []
    with open(file_path, "r") as file:
        current_entry = {}
        
        for line in file:
            line = line.strip()
            if line.startswith("fen"):
                current_entry["fen"] = line.split(" ", 1)[1]
            elif line.startswith("move"):
                current_entry["move"] = line.split(" ", 1)[1]
            elif line.startswith("score"):
                current_entry["score"] = int(line.split(" ", 1)[1])
                if current_entry["score"] == 32002:
                    print(current_entry["fen"])
            elif line.startswith("ply"):
                current_entry["ply"] = int(line.split(" ", 1)[1])
            elif line.startswith("result"):
                current_entry["result"] = int(line.split(" ", 1)[1])
            elif line == "e":  # End of an entry
                if "fen" in current_entry and not "?" in current_entry["move"]:  # Ensure it's a complete entry
                    data.append((
                        current_entry["fen"],
                        current_entry["move"],
                        current_entry["score"],
                        current_entry["ply"],
                        current_entry["result"]
                    ))
                current_entry = {}  # Reset for next entry

    return data



def extract_to_csv(file_path, output_csv):
    with open(file_path, "r") as file, open(output_csv, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["FEN", "Evaluation"]) #Header

        current_fen = None
        current_score = None

        for line in file:
            line = line.strip()

            if line.startswith("fen"):
                current_fen = line.split(" ", 1)[1]
            elif line.startswith("score"):
                current_score = int(line.split(" ", 1)[1])
            elif line == "e":  # End of an entry
                if current_fen and current_score is not None and current_score != 32002:
                    writer.writerow([current_fen, current_score])
                current_fen, current_score = None, None  # Reset for next entry


# Weird stockfish scores
# constexpr Value VALUE_NONE     = 32002;
# constexpr Value VALUE_INFINITE = 32001;
# constexpr Value VALUE_MATE     = 32000;
# Stockfish convert tool
# Stockfish_Build.exe convert test80-2024-01-jan-2tb7p.min-v2.v6.binpack positions01-jan.plain validate
if __name__ == '__main__':
    extract_to_csv("../raw_data/positions03-mar.plain", "../raw_data/positions03-mar.csv")
