import torch
from torch import nn

from evaluator import ModelEvaluator, compare_models
from dataloader import WildfireDataLoaders, wildfire_transforms
from wildfiredb import WildFireData1, WildFireData2, WildFireData3, WildFireData4
from models import FullyConnectedNetwork, Resnet34Scratch, ResNet18PreTrained


def evaluate_models(model_configs, dataloaders, device="cuda"):

    all_evaluators = []
    
    for model_class, source_name in model_configs:
        for dataloader in dataloaders:
            test_dataloader = dataloader.test_dl
            
            if test_dataloader.dataset.source != source_name:
                continue
            
            model_instance = model_class(source_name=source_name).to(device)
            
            class_name = model_class.__name__
            weights_path = f"model_params/{class_name}.pth"
            
            try:
                print(f"\n{'='*80}")
                print(f"Loading model: {class_name}")
                print(f"{'='*80}")
                
                state_dict = torch.load(weights_path, map_location=device)
                model_instance.load_state_dict(state_dict)
                model_instance.eval()
                
                # Run eval
                evaluator = ModelEvaluator(model_instance, device=device)
                results = evaluator.evaluate_full(
                    test_dataloader, 
                    save_dir=f"results/{class_name}"
                )
                
                all_evaluators.append(evaluator)
                
            except FileNotFoundError:
                print(f"Warning: Model weights not found at {weights_path}")
                print(f"Please train the model first using train.py")
                continue
            except Exception as e:
                print(f"Error loading model {class_name}: {e}")
                continue
    
    return all_evaluators


def compare_all_models(evaluators, test_dataloader):
    """Compare all models."""
    if len(evaluators) > 1:
        print(f"\n{'='*80}")
        print("COMPARING ALL MODELS")
        print(f"{'='*80}\n")
        compare_models(evaluators, test_dataloader, save_dir="results/comparison")
    else:
        print("\nNeed at least 2 models to compare.")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    test_dataloaders = [
        WildfireDataLoaders([WildFireData1()], wildfire_transforms),
        # WildfireDataLoaders([WildFireData1(), WildFireData2(),
        #                      WildFireData3(), WildFireData4()], wildfire_transforms),
    ]
    
    models_to_evaluate = [
        (FullyConnectedNetwork, "jafar_2023"),
        (Resnet34Scratch, "jafar_2023"),
        # (ResNet18PreTrained, "jafar_2023"),
    ]
    
    # Evaluate all models
    evaluators = evaluate_models(models_to_evaluate, test_dataloaders, device=device)
    
    if evaluators:
        test_dl = test_dataloaders[0].test_dl
        compare_all_models(evaluators, test_dl)
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)
    print(f"Results saved in 'results/' directory")