"""
Solar Panel Efficiency Prediction
Main entry point for the machine learning pipeline.
"""
import argparse
import warnings
from pathlib import Path

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

from src.utils.config import load_config
from src.utils.logger import setup_logger, get_logger
from train import SolarEfficiencyPipeline

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Solar Panel Efficiency Prediction ML Pipeline"
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--train-data',
        type=str,
        help='Path to training data (overrides config)'
    )
    
    parser.add_argument(
        '--test-data',
        type=str,
        help='Path to test data (overrides config)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--models',
        nargs='+',
        default=['all'],
        help='Models to train (all, traditional, ensemble, neural, or specific model names)'
    )
    
    parser.add_argument(
        '--optimize',
        action='store_true',
        help='Enable hyperparameter optimization'
    )
    
    parser.add_argument(
        '--feature-engineering',
        action='store_true',
        default=True,
        help='Enable advanced feature engineering (default: True)'
    )
    
    parser.add_argument(
        '--no-feature-engineering',
        action='store_true',
        help='Disable feature engineering'
    )
    
    parser.add_argument(
        '--feature-selection',
        action='store_true',
        default=True,
        help='Enable feature selection (default: True)'
    )
    
    parser.add_argument(
        '--no-feature-selection',
        action='store_true',
        help='Disable feature selection'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Handle conflicting feature engineering options
    if args.no_feature_engineering:
        enable_feature_engineering = False
    else:
        enable_feature_engineering = args.feature_engineering
    
    # Handle conflicting feature selection options
    if args.no_feature_selection:
        enable_feature_selection = False
    else:
        enable_feature_selection = args.feature_selection
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logger(
        log_level=log_level,
        log_file=f"{args.output_dir}/logs/pipeline.log"
    )
    
    logger = get_logger(__name__)
    logger.info("Starting Solar Panel Efficiency Prediction Pipeline")
    
    try:
        # Load configuration
        config = load_config(args.config)
        logger.info(f"Configuration loaded from: {args.config}")
        
        # Override config with command line arguments
        if args.train_data:
            config._config['data']['train_file'] = args.train_data
        if args.test_data:
            config._config['data']['test_file'] = args.test_data
        
        # Create output directories
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "plots").mkdir(exist_ok=True)
        (output_path / "models").mkdir(exist_ok=True)
        (output_path / "reports").mkdir(exist_ok=True)
        (output_path / "logs").mkdir(exist_ok=True)
        
        # Initialize and run pipeline
        pipeline = SolarEfficiencyPipeline(
            config=config,
            output_dir=args.output_dir,
            models_to_train=args.models,
            enable_optimization=args.optimize,
            enable_feature_engineering=enable_feature_engineering,
            enable_feature_selection=enable_feature_selection
        )
        
        # Run the complete pipeline
        results = pipeline.run()
        
        logger.info("Pipeline completed successfully!")
        logger.info(f"Results saved to: {args.output_dir}")
        
        # Print summary
        if results:
            best_model = results['best_model']
            best_score = results['best_score']
            logger.info(f"Best model: {best_model} (R² = {best_score:.4f})")
            
            print("\n" + "="*60)
            print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*60)
            print(f"✅ Best Model: {best_model}")
            print(f"✅ Best R² Score: {best_score:.4f}")
            print(f"✅ Models Trained: {len(results['all_model_results'])}")
            print(f"✅ Results saved to: {args.output_dir}")
            print("="*60)
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()