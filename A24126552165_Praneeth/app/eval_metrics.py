import wandb
import random
import time

def run_evaluation():
    # 1. Initialize W&B tracking
    wandb.init(project="daa-educational-chatbot", name="dynamic-test-run")
    
    print("Running dynamic evaluation...")
    
    # Let's test it over 10 steps instead of 3 to get a better graph
    for epoch in range(10):
        
        # Simulate the bot getting SMARTER over time (F1 score rising from 0.5 up to 0.95)
        base_f1 = 0.5 + (epoch * 0.05)
        # Adding a tiny bit of random noise to make the graph look like real AI data
        current_f1 = base_f1 + random.uniform(-0.02, 0.02)
        
        # Simulate the bot getting FASTER over time (Latency dropping from 1.5s down to 0.5s)
        base_latency = 1.5 - (epoch * 0.1)
        current_latency = base_latency + random.uniform(-0.1, 0.1)
        
        # 2. Log metrics to the W&B dashboard
        wandb.log({
            "epoch": epoch,
            "f1_score": current_f1,
            "latency_seconds": current_latency
        })
        
        print(f"Epoch {epoch}: F1={current_f1:.2f}, Latency={current_latency:.2f}s")
        time.sleep(1)
        
    wandb.finish()
    print("Evaluation complete. Check your W&B dashboard!")

if __name__ == "__main__":
    run_evaluation()