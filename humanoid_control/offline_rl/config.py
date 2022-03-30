from ml_collections import ConfigDict
from humanoid_control import observables

def prepend_model(name):
    return "humanoid_control.distillation.model." + name

def get_config(config_string):
    configs = {
        'mlp_time_index': ConfigDict({
            'constructor': prepend_model('MlpPolicy'),
            'config': ConfigDict(dict(
                n_layers=3,
                layer_size=1024,
                activation_fn='torch.nn.Tanh',
                squash_output=True,
                observables=observables.TIME_INDEX_OBSERVABLES
                ))
            }),
        'mlp_reference': ConfigDict({
            'constructor': prepend_model('MlpPolicy'),
            'config': ConfigDict(dict(
                n_layers=3,
                layer_size=1024,
                activation_fn='torch.nn.Tanh',
                squash_output=True,
                observables=observables.HIGH_LEVEL_OBSERVABLES
                ))
            }),
        'hierarchical_mlp': ConfigDict({
            'constructor': prepend_model('HierarchicalMlpPolicy'),
            'config': ConfigDict(dict(
                embed_size=20,
                ref_encoder_n_layers=2,
                ref_encoder_layer_size=1024,
                decoder_n_layers=3,
                decoder_layer_size=1024,
                embedding_kl_weight=0.1,
                activation_fn='torch.nn.Tanh',
                squash_output=True,
                observables=observables.HIERARCHICAL_OBSERVABLES
                ))
            }),
        'hierarchical_rnn': ConfigDict({
            'constructor': prepend_model('HierarchicalRnnPolicy'),
            'config': ConfigDict(dict(
                embed_size=20,
                ref_encoder_n_layers=2,
                ref_encoder_layer_size=1024,
                decoder_n_layers=3,
                decoder_layer_size=1024,
                layer_norm=True,
                embedding_kl_weight=0.1,
                embedding_correlation=0.95,
                seq_steps=30,
                activation_fn='torch.nn.Tanh',
                squash_output=True,
                observables=observables.HIERARCHICAL_OBSERVABLES
                ))
            }),
        'gpt': ConfigDict({
            'constructor': prepend_model('GPTPolicy'),
            'config': ConfigDict(dict(
                block_size=16,
                n_layer=4,
                n_head=4,
                n_embd=768,
                embd_pdrop=0.1,
                resid_pdrop=0.1,
                attn_pdrop=0.1,
                weight_decay=0.1,
                observables=observables.HIGH_LEVEL_OBSERVABLES_SANS_REFERENCE
                ))
            }),
        #'hierarchical_gpt': ConfigDict({
        #    'constructor': prepend_model('HierarchicalGPTPolicy'),
        #    'config': ConfigDict(dict(
        #        block_size=16,
        #        n_layer=4,
        #        n_head=4,
        #        n_embd=768,
        #        embd_pdrop=0.1,
        #        resid_pdrop=0.1,
        #        attn_pdrop=0.1,
        #        observables=observables.HIERARCHICAL_OBSERVABLES
        #        ))
        #    }),
        #'mixed_hierarchical': ConfigDict({
        #    'constructor': prepend_model('MixedHierarchicalPolicy'),
        #    'config': ConfigDict(dict(
        #        block_size=16,
        #        n_layer=4,
        #        n_head=4,
        #        n_embd=768,
        #        embd_pdrop=0.1,
        #        resid_pdrop=0.1,
        #        attn_pdrop=0.1,
        #        embed_size=20,
        #        layer_norm=True,
        #        decoder_n_layers=3,
        #        decoder_layer_size=1024,
        #        kl_weight=0.002,
        #        activation_fn='torch.nn.Tanh',
        #        squash_output=True,
        #        action_decoder_load_path="",
        #        observables=observables.MIXED_HIERARCHICAL_OBSERVABLES
        #        ))
        #    }),
        #'hierarchical': ConfigDict({
        #    'constructor': prepend_model('HierarchicalPolicy'),
        #    'config': ConfigDict(dict(
        #        embed_size=20,
        #        ref_encoder_n_layers=2,
        #        ref_encoder_layer_size=1024,
        #        stand_encoder_n_layers=2,
        #        stand_encoder_layer_size=1024,
        #        decoder_n_layers=3,
        #        decoder_layer_size=1024,
        #        stand_vf_n_layers=3,
        #        stand_vf_layer_size=1024,
        #        layer_norm=True,
        #        embedding_kl_weight=0.1,
        #        embedding_correlation=0.95,
        #        seq_steps=30,
        #        activation_fn='torch.nn.Tanh',
        #        squash_output=False,
        #        observables=observables.HYBRID_HIERARCHICAL_OBSERVABLES
        #    ))
        #})
    }

    return configs[config_string]