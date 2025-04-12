# Core Modules

The core modules of FatPy provide the fundamental analytical methods for fatigue life evaluation.

## Available Modules

- [Core](core/index.md): Methods based on stress-based approaches
- [Data Parsing](data_parsing.md): Tools for parsing and processing data
- [Utilities](utilities.md): Helper functions and utilities for various tasks

## Module Structure

*TO BE IMPLEMENTED*

## Component Diagram

```mermaid
graph TB
    Main[FatPy Main Interface]

      subgraph MainComponents[Main Components]
        direction LR
        Core[Core]
        DataParsing[_ImportExport]
        Utils[Utilities]
      end

      Main --> MainComponents

      subgraph CoreComponents[Core Components]
        direction LR
        StressLife[Stress-Life]
        StrainLife[Strain-Life]
        EnergyLife[Energy-Life]
        _Template[Template]
      end

      Core --> CoreComponents

      subgraph SubComponents1[Stress-Life Components]
        direction LR
        BaseMethods1[Base Methods]
        CorrectionMethod1[Correction Method]
        Decomposition1[Decomposition]
        DamageParam1[Damage Parameters]
      end

      subgraph SubComponents2[Strain-Life Components]
        direction LR
        BaseMethods2[Base Methods]
        CorrectionMethod2[Correction Method]
        Decomposition2[Decomposition]
        DamageParam2[Damage Parameters]
      end

      subgraph SubComponents3[Energy-Life Components]
        direction LR
        BaseMethods3[Base Methods]
        CorrectionMethod3[Correction Method]
        Decomposition3[Decomposition]
        DamageParam3[Damage Parameters]
      end

      subgraph Teplate[_Template Components]
        direction LR
        _BaseMethods[Base Methods]
        _CorrectionMethod[Correction Method]
        _Decomposition[Decomposition]
        _DamageParam[Damage Parameters]
      end

      StressLife --> SubComponents1
      StrainLife --> SubComponents2
      EnergyLife --> SubComponents3
      _Template --> Teplate


      subgraph DataParsingComponents[Data Parsing Components]
        direction LR
        UserInput[User Input]
        Material[Material]
        FeModel[FE Model]
        Loads[Loads]
      end

      DataParsing --> DataParsingComponents

      subgraph UtilComponents[Util-Components]
        direction LR
        Transformation[Transformation]
        PreProcessing[Pre-Processing]
        PostProcessing[Post-Processing]
      end

      Utils --> UtilComponents


```

## Theory Reference

For a deeper understanding of the methods and their theoretical background, refer to the [Theory Reference](../theory/index.md).
