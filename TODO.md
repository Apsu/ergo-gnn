# TODO List for ERGO-GNN Project

## High Priority

### Configuration Issues
- [ ] **Semantic threshold too high**: Current threshold of 0.85 prevents semantic edges
  - Observed similarities typically range from 0.5-0.84
  - Consider lowering default to 0.7 or 0.75
  - Make threshold configurable via command line args

### Missing Functionality
- [ ] **Logging System**: No logs are currently being written to the `logs/` directory
  - Add file logging handlers to all scripts
  - Consider structured logging for training metrics
  - Add rotation/retention policies for log files

- [ ] **Data Processing Pipeline**: The `data/processed/` directory is unused
  - Decide if preprocessing is needed (e.g., pre-computed embeddings)
  - Either implement processing pipeline or remove from config
  - Update documentation accordingly

### Missing Dependencies
- [ ] **Create requirements.txt** with all dependencies:
  ```
  torch
  torch-geometric
  transformers
  sentence-transformers
  openai
  numpy
  matplotlib
  seaborn
  tqdm
  pydantic
  ```

### Script Issues
- [ ] **Fix script references** in documentation:
  - `evaluate_gnn.py` referenced but doesn't exist (use `evaluate_unified.py`)
  - `use_gnn_context.py` referenced but doesn't exist

## Medium Priority

### Code Improvements
- [ ] **Add comprehensive error handling** for API failures in data generation
- [ ] **Implement data validation** for loaded conversations
- [ ] **Add model versioning** to track experiments
- [ ] **Create inference script** (`use_gnn_context.py`) for production use

### Performance Optimizations
- [ ] **Implement embedding caching** to avoid recomputing
- [ ] **Add batch processing** for SimpleGraphManager operations
- [ ] **Optimize graph construction** for very long conversations

## Low Priority

### Documentation
- [ ] **Update examples** in README to use actual generated data
- [ ] **Add API documentation** for main classes
- [ ] **Create troubleshooting guide** for common issues
- [ ] **Add performance benchmarks** and scaling guide

### Testing
- [ ] **Add unit tests** for core components
- [ ] **Create integration tests** for full pipeline
- [ ] **Add regression tests** for model performance

## Future Enhancements

### Features
- [ ] **Multi-turn context windows** for better coherence
- [ ] **Conversation summarization** for very long histories
- [ ] **Dynamic graph pruning** for memory efficiency
- [ ] **Model explainability** tools for attention analysis

### Infrastructure
- [ ] **MLflow/W&B integration** for experiment tracking
- [ ] **Docker containerization** for deployment
- [ ] **REST API** for model serving
- [ ] **Distributed training** support for large datasets

## Completed ✅

- [x] LLM-supervised learning integration
- [x] Directory structure reorganization
- [x] Default paths in config.py
- [x] Interruption handling for data generation
- [x] Checkpoint/resume functionality
- [x] MPS device support for evaluation
- [x] Unified evaluation script
- [x] Remove backwards compatibility code
- [x] Standardize on PyTorch Geometric

## Notes

- The project is functional but needs production hardening
- Current focus should be on missing functionality (logging, requirements.txt)
- Data processing pipeline decision needed before v1.0