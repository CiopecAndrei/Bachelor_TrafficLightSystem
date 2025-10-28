import joblib
import pandas as pd

def load_model(model_path):
    return joblib.load(model_path)

def get_manual_input():
    print("Enter number of cars per lane (0-3):")
    cars_south = int(input("South lane cars: "))
    cars_north = int(input("North lane cars: "))
    cars_west = int(input("West lane cars: "))
    cars_east = int(input("East lane cars: "))

    for val in [cars_south, cars_north, cars_west, cars_east]:
        if val < 0 or val > 3:
            raise ValueError("Number of cars must be between 0 and 3")

    data = {
        'cars_south': [cars_south],
        'cars_north': [cars_north],
        'cars_west': [cars_west],
        'cars_east': [cars_east]
    }
    return pd.DataFrame(data)

def decide_light_phase(cars_south, cars_north, cars_west, cars_east):
    sn_total = cars_south + cars_north
    ew_total = cars_west + cars_east
    if sn_total > ew_total:
        return "SN (South-North lanes green)"
    elif ew_total > sn_total:
        return "EW (East-West lanes green)"
    else:
        # Tie-breaker, you can choose any
        return "SN (South-North lanes green) (tie-breaker)"

def predict_bucket(model, features_df):
    prediction = model.predict(features_df)[0]
    bucket_names = {0: "Short green time", 1: "Medium green time", 2: "Long green time"}
    return prediction, bucket_names[prediction]

def main():
    model_path = r"C:\Users\ciope\Desktop\Licenta\prototypes\Train_times_ML\models\decision_tree_model.pkl"
    model = load_model(model_path)
    
    features_df = get_manual_input()
    cars_south = features_df['cars_south'].values[0]
    cars_north = features_df['cars_north'].values[0]
    cars_west = features_df['cars_west'].values[0]
    cars_east = features_df['cars_east'].values[0]

    phase = decide_light_phase(cars_south, cars_north, cars_west, cars_east)
    bucket_num, bucket_name = predict_bucket(model, features_df)
    
    print(f"\nPredicted green light phase: {phase}")
    print(f"Predicted green time bucket: {bucket_num} ({bucket_name})")

if __name__ == "__main__":
    main()
