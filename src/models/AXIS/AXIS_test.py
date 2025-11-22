import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Tuple, List, Dict, Any, Union, TypedDict, Optional
from dataclasses import dataclass
import sys
from datetime import datetime
import yaml

from src.models.AXIS.dataset import AXISAnomalyQADataset
from src.models.AXIS.AXIS import AXISCombinedModel
from experiments.configs.axis_config import AXISConfig, default_config

import warnings
warnings.filterwarnings("ignore")

# Configuration
TOTAL_BATCHES = 70  # Number of batches to test


class TeeLogger:
    """A simple logger that outputs to both console and file."""
    
    def __init__(self, log_file_path: str):
        self.log_file_path = log_file_path
        self.log_file = None
        self._open_log_file()
    
    def _open_log_file(self):
        """Open the log file for writing."""
        try:
            self.log_file = open(self.log_file_path, 'w', encoding='utf-8')
            self.log_file.write(f"=== AXIS Test Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")
            self.log_file.flush()
        except Exception as e:
            print(f"Warning: Could not open log file {self.log_file_path}: {e}")
            self.log_file = None
    
    def print(self, *args, **kwargs):
        """Print to both console and log file."""
        # Print to console
        print(*args, **kwargs)
        
        # Print to log file if available
        if self.log_file:
            try:
                # Convert all arguments to strings and join them
                message = ' '.join(str(arg) for arg in args)
                self.log_file.write(message + '\n')
                self.log_file.flush()
            except Exception as e:
                print(f"Warning: Could not write to log file: {e}")
    
    def close(self):
        """Close the log file."""
        if self.log_file:
            try:
                self.log_file.write(f"\n=== Log ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
                self.log_file.close()
                self.log_file = None
            except Exception:
                pass

# Global logger instance
logger = None

@dataclass
class AnalysisItem:
    question: str
    answer: str
    start_index: int
    end_index: int
    question_type: str

class BatchItem(TypedDict):
    time_series: List[float]
    analysis_data: Union[List[Dict[str, str]], Dict[str, str]]

class CollatedBatch(TypedDict):
    padded_sequences: torch.Tensor
    attention_masks: torch.Tensor
    questions: List[str]
    answers: List[str]
    start_indices: List[int]
    end_indices: List[int]
    question_types: List[str]

def process_analysis_data(analysis_data: Union[List[Dict[str, str]], Dict[str, str]]) -> List[AnalysisItem]:
    if isinstance(analysis_data, dict):
        return [AnalysisItem(
            question=analysis_data['question'],
            answer=analysis_data['answer'],
            start_index=analysis_data['window_range']['start'],
            end_index=analysis_data['window_range']['end'],
            question_type=analysis_data.get('question_type', 'unknown')
        )]
    
    return [AnalysisItem(
        question=item['question'],
        answer=item['answer'],
        start_index=item['window_range']['start'],
        end_index=item['window_range']['end'],
        question_type=item.get('question_type', 'unknown')
    ) for item in analysis_data]

def collate_fn(batch: List[BatchItem]) -> CollatedBatch:
    if not batch:
        raise ValueError("Empty batch received")

    # Convert time series to tensors (1D per sample, consistent with Moirai_main_Zero2)
    time_series_tensors: List[torch.Tensor] = [
        torch.tensor(item['time_series'], dtype=torch.bfloat16)
        for item in batch
    ]

    # Process analysis data
    all_analysis_items: List[AnalysisItem] = []
    repeated_time_series: List[torch.Tensor] = []
    
    for ts_tensor, item in zip(time_series_tensors, batch):
        analysis_items = process_analysis_data(item['analysis_data'])
        all_analysis_items.extend(analysis_items)
        repeated_time_series.extend([ts_tensor] * len(analysis_items))

    # Pad sequences to (B, max_seq_len)
    padded_sequences = torch.nn.utils.rnn.pad_sequence(
        repeated_time_series,
        batch_first=True,
        padding_value=0.0
    )

    # Create attention masks
    sequence_lengths = [seq.size(0) for seq in repeated_time_series]
    max_sequence_length = padded_sequences.size(1)
    attention_masks = torch.zeros(len(repeated_time_series), max_sequence_length, dtype=torch.bool)
    
    for idx, length in enumerate(sequence_lengths):
        attention_masks[idx, :length] = True

    return CollatedBatch(
        padded_sequences=padded_sequences,
        attention_masks=attention_masks,
        questions=[item.question for item in all_analysis_items], 
        answers=[item.answer for item in all_analysis_items],
        start_indices=[item.start_index for item in all_analysis_items],
        end_indices=[item.end_index for item in all_analysis_items],
        question_types=[item.question_type for item in all_analysis_items]
    )

def save_question_result(question_idx: int, result_data: Dict[str, Any], save_dir: str, suffix: str = ""):
    """Save individual question result as YAML file."""
    try:
        os.makedirs(save_dir, exist_ok=True)
        
        # Create filename with zero-padding for proper sorting
        filename = f"question_{question_idx:06d}{suffix}.yaml"
        filepath = os.path.join(save_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(result_data, f, default_flow_style=False, allow_unicode=True, indent=2)
            
        return filepath
    except Exception as e:
        print(f"Error saving question {question_idx}: {str(e)}")
        return None

def test_model_forward(model: nn.Module,
                      batch: Dict,
                      device: torch.device) -> Dict[str, Any]:
    """Forward pass and generation using AXISCombinedModel."""
    model.eval()
    
    with torch.amp.autocast("cuda"):
        with torch.no_grad():
            loss = model(
                padded_sequences=batch['padded_sequences'].to(device),
                attention_masks=batch['attention_masks'].to(device),
                questions=batch['questions'],
                answers=batch['answers'],
                start_indices=batch['start_indices'],
                end_indices=batch['end_indices']
            )
            # Direct generation through ts_pretrain_model and axis.generate to avoid CombinedModel.generate return value inconsistency
            try:
                local_embeddings = model.ts_pretrain_model(
                    batch['padded_sequences'].to(device),
                    mask=batch['attention_masks'].to(device)
                )
                responses = model.axis.generate(
                    local_embeddings=local_embeddings,
                    time_series=batch['padded_sequences'].to(device),
                    questions=batch['questions'],
                    answers=batch['answers'],
                    start_indices=batch['start_indices'],
                    end_indices=batch['end_indices']
                )
                # Ablations
                # responses_wo_windows = model.axis.generate(
                #     local_embeddings=local_embeddings,
                #     time_series=batch['padded_sequences'].to(device),
                #     questions=batch['questions'],
                #     answers=batch['answers'],
                #     start_indices=batch['start_indices'],
                #     end_indices=batch['end_indices'],
                #     ablation_mode="wo_windows"
                # )
                # responses_wo_hint = model.axis.generate(
                #     local_embeddings=local_embeddings,
                #     time_series=batch['padded_sequences'].to(device),
                #     questions=batch['questions'],
                #     answers=batch['answers'],
                #     start_indices=batch['start_indices'],
                #     end_indices=batch['end_indices'],
                #     ablation_mode="wo_hint"
                # )
            except Exception:
                responses = ["Generation method not available"] * len(batch['questions'])
                # responses_wo_windows = ["Generation method not available"] * len(batch['questions'])
                # responses_wo_hint = ["Generation method not available"] * len(batch['questions'])
    
    return {
        'loss': float(loss.item()),
        'questions': batch['questions'],
        'expected_answers': batch['answers'],
        'generated_responses': responses,
        # 'generated_responses_wo_windows': responses_wo_windows,
        # 'generated_responses_wo_hint': responses_wo_hint,
        'batch_size': len(batch['questions']),
        'start_indices': batch['start_indices'],
        'end_indices': batch['end_indices'],
        'question_types': batch['question_types']
    }

def _sanitize_llm_name(model_name: str) -> str:
    """Sanitize LLM model name for safe filesystem usage."""
    # Replace slashes, spaces and other non filename-friendly chars with underscore
    return re.sub(r"[^A-Za-z0-9_.\-]+", "_", model_name)


def _get_llm_subdir(config: AXISConfig) -> str:
    """Return sanitized LLM subdirectory name based on config."""
    try:
        model_name = getattr(config.llm_config, "model_name", "unknown_llm")
    except Exception:
        model_name = "unknown_llm"
    return _sanitize_llm_name(model_name)

def load_checkpoint_for_test(model: nn.Module,
                             config: AXISConfig,
                             device: torch.device) -> bool:
    """Load checkpoint in AXIS format (prioritize Accelerate format)."""
    global logger

    # 1) Prioritize Accelerate saved directory
    candidates: List[str] = []
    llm_subdir = _get_llm_subdir(config)
    base_dir = os.path.join(config.checkpoint_dir, llm_subdir)
    best_dir = os.path.join(base_dir, f"{config.model_prefix}_best_accelerate", "model_optimizer.pth")
    step_dir = os.path.join(base_dir, f"{config.model_prefix}_step_accelerate", "model_optimizer.pth")
    candidates.extend([best_dir, step_dir])

    # 2) Compatible with legacy paths
    if hasattr(config, 'load_path') and config.load_path:
        candidates.append(config.load_path)

    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            checkpoint = torch.load(path, map_location=device)
        except Exception as e:
            if logger:
                logger.print(f"Failed to load {path}: {e}")
            else:
                print(f"Failed to load {path}: {e}")
            continue

        # New format
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state = checkpoint['model_state_dict']
            unwrapped = model
            # ts_pretrain_model
            if 'ts_pretrain_model' in state:
                unwrapped.ts_pretrain_model.load_state_dict(state['ts_pretrain_model'])
                if logger:
                    logger.print("Loaded ts_pretrain_model parameters")
                else:
                    print("Loaded ts_pretrain_model parameters")
            # axis trainable parameters
            if 'moirai_trainable' in state:
                axis_trainable_state_dict = state['moirai_trainable']
                for name, param in unwrapped.axis.named_parameters():
                    if param.requires_grad and name in axis_trainable_state_dict:
                        param.data.copy_(axis_trainable_state_dict[name])
                if logger:
                    logger.print(f"Loaded {len(axis_trainable_state_dict)} trainable axis parameters")
                else:
                    print(f"Loaded {len(axis_trainable_state_dict)} trainable axis parameters")
            if logger:
                logger.print(f"Checkpoint loaded from {path}")
            else:
                print(f"Checkpoint loaded from {path}")
            return True

        # Legacy format (direct state_dict)
        try:
            model.load_state_dict(checkpoint, strict=False)
            if logger:
                logger.print(f"Legacy checkpoint loaded from {path}")
            else:
                print(f"Legacy checkpoint loaded from {path}")
            return True
        except Exception:
            pass

    if logger:
        logger.print("No available AXIS checkpoint found, using randomly initialized weights for testing.")
    else:
        print("No available AXIS checkpoint found, using randomly initialized weights for testing.")
    return False

def run_simple_test():
    """Run a simple test of the model."""
    global logger
    
    # Initialize logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = f"experiments/logs/AXIS/axis_test_{timestamp}.txt"
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    logger = TeeLogger(log_file_path)
    
    try:
        # Load configuration
        config = default_config
        
        # Set device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.print(f"Using device: {device}")
        
        # Initialize model
        logger.print("Initializing AXISCombinedModel...")
        model = AXISCombinedModel(config).to(device)
        
        # Try to load checkpoint if available
        load_checkpoint_for_test(model, config, device)
        
        # Load test data
        logger.print("Loading test data...")
        test_dataset = AXISAnomalyQADataset(config.test_data_path, split="test", train_ratio=0.02)
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,  # Small batch size for testing
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,  # Avoid multiprocessing issues in testing
        )
        
        logger.print(f"Test dataset size: {len(test_dataset)}")
        
        # Create save directory for YAML files
        llm_subdir = _get_llm_subdir(config)
        save_dir = f"./experiments/logs/AXIS/{llm_subdir}/results_{timestamp}"
        os.makedirs(save_dir, exist_ok=True)
        
        # Run test on first few batches
        logger.print("\nRunning model tests...")
        logger.print(f"Results will be saved to: {save_dir}")
        
        total_questions = 0
        total_loss = 0.0
        total_batches = 0
        
        for i, batch in enumerate(test_loader):
            if i >= TOTAL_BATCHES:  # Only test first TOTAL_BATCHES batches
                break
                
            logger.print(f"\n--- Testing Batch {i+1} ---")
            logger.print(f"Batch questions: {len(batch['questions'])}")
            logger.print(f"Time series shape: {batch['padded_sequences'].shape}")
            
            try:
                results = test_model_forward(model, batch, device)
                total_loss += results['loss']
                total_batches += 1
                
                logger.print(f"Loss: {results['loss']:.4f}")
                
                # Save each question result as separate YAML file
                for j in range(len(results['questions'])):
                    question_data = {
                        'batch_id': i + 1,
                        'question_in_batch': j + 1,
                        'global_question_id': total_questions + j + 1,
                        'timestamp': datetime.now().isoformat(),
                        'question': results['questions'][j],
                        'expected_answer': results['expected_answers'][j],
                        'generated_response': results['generated_responses'][j] if isinstance(results['generated_responses'], list) else str(results['generated_responses']),
                        'question_type': results['question_types'][j],
                        'window_range': {
                            'start': int(batch['start_indices'][j]),
                            'end': int(batch['end_indices'][j])
                        },
                        'model_config': {
                            'model_name': 'AXIS',
                            'device': str(device),
                            'batch_size': len(batch['questions']),
                            'checkpoint_dir': config.checkpoint_dir,
                            'model_prefix': config.model_prefix
                        },
                        'model_outputs': {
                            'loss': results['loss']
                        }
                    }
                    
                    saved_path = save_question_result(total_questions + j + 1, question_data, save_dir)
                    if saved_path:
                        logger.print(f"Saved question {total_questions + j + 1} to {saved_path}")

                    # Save ablation: w/o windows
                    # question_data_wo_windows = dict(question_data)
                    # question_data_wo_windows['generated_response'] = results['generated_responses_wo_windows'][j] if isinstance(results['generated_responses_wo_windows'], list) else str(results['generated_responses_wo_windows'])
                    # saved_path_wo_windows = save_question_result(total_questions + j + 1, question_data_wo_windows, save_dir, suffix="_wo_windows")
                    # if saved_path_wo_windows:
                    #     logger.print(f"Saved question {total_questions + j + 1} (w/o windows) to {saved_path_wo_windows}")

                    # # Save ablation: w/o Hint
                    # question_data_wo_hint = dict(question_data)
                    # question_data_wo_hint['generated_response'] = results['generated_responses_wo_hint'][j] if isinstance(results['generated_responses_wo_hint'], list) else str(results['generated_responses_wo_hint'])
                    # saved_path_wo_hint = save_question_result(total_questions + j + 1, question_data_wo_hint, save_dir, suffix="_wo_hint")
                    # if saved_path_wo_hint:
                    #     logger.print(f"Saved question {total_questions + j + 1} (w/o hint) to {saved_path_wo_hint}")
                
                total_questions += len(results['questions'])
                
                # Print first question and answer pair
                if results['questions']:
                    logger.print(f"Sample Question: {results['questions'][0]}")
                    logger.print(f"Expected Answer: {results['expected_answers'][0]}")
                    logger.print(f"Generated Response: {results['generated_responses'][0] if isinstance(results['generated_responses'], list) else str(results['generated_responses'])}")
                    
            except Exception as e:
                logger.print(f"Error in batch {i+1}: {str(e)}")
                import traceback
                error_traceback = traceback.format_exc()
                logger.print(error_traceback)
        
        # Save summary
        summary_data = {
            'test_summary': {
                'timestamp': datetime.now().isoformat(),
                'total_batches_tested': total_batches,
                'total_questions_processed': total_questions,
                'average_loss': total_loss / max(total_batches, 1),
                'model_config': {
                    'model_name': 'AXIS',
                    'device': str(device),
                    'batch_size': 2,
                    'checkpoint_dir': config.checkpoint_dir,
                    'model_prefix': config.model_prefix,
                    'test_data_path': config.test_data_path
                },
                'test_statistics': {
                    'avg_questions_per_batch': total_questions / max(total_batches, 1),
                    'total_batches_configured': TOTAL_BATCHES
                }
            }
        }
        
        summary_path = os.path.join(save_dir, "test_summary.yaml")
        with open(summary_path, 'w', encoding='utf-8') as f:
            yaml.dump(summary_data, f, default_flow_style=False, allow_unicode=True, indent=2)
        
        logger.print("\nTest completed!")
        logger.print(f"Total questions processed: {total_questions}")
        logger.print(f"Average loss: {total_loss / max(total_batches, 1):.4f}")
        logger.print(f"Results saved to: {save_dir}")
        logger.print(f"Summary saved to: {summary_path}")
        logger.print(f"Log saved to: {log_file_path}")
        
    except Exception as e:
        if logger:
            logger.print(f"Fatal error: {str(e)}")
            import traceback
            error_traceback = traceback.format_exc()
            logger.print(error_traceback)
        else:
            print(f"Fatal error: {str(e)}")
            import traceback
            traceback.print_exc()
    
    finally:
        # Close logger
        if logger:
            logger.close()

if __name__ == "__main__":
    run_simple_test()

